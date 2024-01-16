import glob
import os
from itertools import chain

import SimpleITK as sitk
import numpy as np

import pyramid3dunet.augment.transforms as transforms
from pyramid3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats, zoom
from pyramid3dunet.unet3d.utils import get_logger
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import math
logger = get_logger('NIIDataset')

class AbstractNIIDataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the NII files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, raw_file_path, label_file_path, 
                 phase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 weight_file_path=None,
                 global_normalization=True):
        """
        :param file_path: path to NII file containing raw data as well as labels and per pixel weights (optional)
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): NII internal path to the raw dataset
        :param label_internal_path (str or list): NII internal path to the label dataset
        :param weight_file_path (str or list): NII internal path to the per pixel weights
        """
        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phase = phase
        self.file_path = raw_file_path
        self.raw = self.create_nii_file(raw_file_path)
        if global_normalization:
            stats = calculate_stats(self.raw)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase == 'test':
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None
            self.weight_map = None
            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                if self.raw.ndim == 4:
                    channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in self.raw]
                    self.raw = np.stack(channels)
                else:
                    self.raw = np.pad(self.raw, pad_width=pad_width, mode='reflect')
        else:
            if label_file_path is None: 
                self.label = None
                self.weight_map = None
            else:
                # create label/weight transform only in train/val phase
                self.label_transform = self.transformer.label_transform()
                self.label = self.create_nii_file(label_file_path)   
                # self.label = 1 - (self.label > 0).astype(float)
                if weight_file_path is not None:
                    # look for the weight map in the raw file
                    self.weight_map = self.create_nii_file(weight_file_path)
                    self.weight_transform = self.transformer.weight_transform()
                else:
                    self.weight_map = None
                self._check_volume_sizes(self.raw, self.label)

        # build slice indices for raw and label data sets
        print('loader: ', self.raw.shape)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        if self.phase == 'test' or self.label is None:
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])
            if self.weight_map is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            return raw_patch_transformed, label_patch_transformed 

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        raw_dir = dataset_config.get('raw_dir', 'raw')
        label_dir = dataset_config.get('label_dir', None)
        weight_dir = dataset_config.get('weight_dir', None)
        raw_file_paths = cls.traverse_nii_paths(file_paths, raw_dir)
        if label_dir is not None:
            label_file_paths = cls.traverse_nii_paths(file_paths, label_dir)
        else:
            label_file_paths = [None for _ in raw_file_paths]
        if weight_dir is not None:
            weight_file_paths = cls.traverse_nii_paths(file_paths, weight_dir)
        else:
            weight_file_paths = [None for _ in raw_file_paths]


        datasets = []
        sid = 0
        for raw_file_path, label_file_path, weight_file_path in zip(raw_file_paths, label_file_paths, weight_file_paths):
            try:
                logger.info(f'Loading {phase} set from: {raw_file_path}...')
                dataset = cls(raw_file_path=raw_file_path,
                              label_file_path =label_file_path,
                              weight_file_path=weight_file_path,
                              phase=phase, 
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
                sid += 1
                # only one sample
                # break
            except Exception:
                logger.error(f'Skipping {phase} set: {raw_file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_nii_paths(roots, subfolder):
        assert isinstance(roots, list)
        results = []
        for root_dir in roots:
            target_dir = root_dir + '/' + subfolder
            if os.path.isdir(target_dir):
                # if file path is a directory take all NII files in that directory
                iters = [sorted(glob.glob(os.path.join(target_dir, '*.nii.gz'))) ]
                # print(iters)
                for fp in chain(*iters):
                    results.append(fp)
            else:
                print('root_dir/subfolder should be a directory: root_dir/subfolder/data.nii')
        return sorted(results)




class StandardNIIDataset(AbstractNIIDataset):
    """
    Implementation of the NII dataset which loads the data from all of the NII files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, raw_file_path, label_file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 weight_file_path=None, global_normalization=True):
        super().__init__(raw_file_path=raw_file_path,
                         label_file_path=label_file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         weight_file_path=weight_file_path,
                         global_normalization=global_normalization)
        slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices
        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')
    
    @staticmethod
    def create_nii_file(file_path):
        image = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(image)
## use a coarse segmentation to extract the patches we want to pay more attention.
class CoarseSegmentedNIIDataset(AbstractNIIDataset):
    """
    Implementation of the NII dataset which loads the data from all of the NII files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, raw_file_path, label_file_path, cmask_file_path, phase, slice_builder_config, transformer_config, mirror_padding=(16, 32, 32),
                 weight_file_path=None, global_normalization=True, dilation_iteration = 5):
        super().__init__(raw_file_path=raw_file_path,
                         label_file_path=label_file_path,
                         phase=phase,
                         slice_builder_config=slice_builder_config,
                         transformer_config=transformer_config,
                         mirror_padding=mirror_padding,
                         weight_file_path=weight_file_path,
                         global_normalization=global_normalization)
        ## re-write slice-builder part

        cmask = self.create_nii_file(cmask_file_path)

        cmask = binary_dilation(cmask > 0.5, iterations = dilation_iteration).astype(float)
        ## we have a zoom process
        if cmask.shape != self.raw.shape:
            cmask = zoom(cmask, target_shape=self.raw.shape)
        cmask[cmask > 0.0] = 1
        ske_mask = skeletonize(cmask > 0)

        if self.label is not None:
            ## in training stage, combine label and prediction to get better training results.
            ske_mask = np.logical_or(skeletonize(self.label > 0), ske_mask)

        slice_builder_config['mask'] = ske_mask
        slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices
        self.weight_slices = slice_builder.weight_slices
        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')        

    
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load slice builder config
        slice_builder_config = phase_config['slice_builder']
        # load files to process
        file_paths = phase_config['file_paths']
        raw_dir = dataset_config.get('raw_dir', 'raw')
        label_dir = dataset_config.get('label_dir', None)
        weight_dir = dataset_config.get('weight_dir', None)
        mask_path = phase_config.get('mask_path', None)
        raw_file_paths = cls.traverse_nii_paths(file_paths, raw_dir)
        if label_dir is not None:
            label_file_paths = cls.traverse_nii_paths(file_paths, label_dir)
        else:
            label_file_paths = [None for _ in raw_file_paths]
        if weight_dir is not None:
            weight_file_paths = cls.traverse_nii_paths(file_paths, weight_dir)
        else:
            weight_file_paths = [None for _ in raw_file_paths]
        if mask_path is not None:
            mask_file_paths = cls.traverse_nii_paths([mask_path], '')
        else:
            mask_file_paths = [None for _ in raw_file_paths]

        datasets = []
        sid = 0
        for raw_file_path, label_file_path, weight_file_path, mask_file_path in zip(raw_file_paths, label_file_paths, weight_file_paths, mask_file_paths):
            try:
                logger.info(f'Loading {phase} set from: {raw_file_path}...')
                dataset = cls(raw_file_path=raw_file_path,
                              label_file_path =label_file_path,
                              weight_file_path=weight_file_path,
                              cmask_file_path = mask_file_path,
                              phase=phase, 
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
                sid += 1
                # only one sample
                # break
            except Exception:
                logger.error(f'Skipping {phase} set: {raw_file_path}', exc_info=True)
        return datasets
    @staticmethod
    def create_nii_file(file_path):
        image = sitk.ReadImage(file_path)
        return sitk.GetArrayFromImage(image)

## If the dataset is too large, use lazy class to load the data from hard disk 
class LazyNIIDataset(ConfigDataset):
    """
    Implementation of the NII dataset which loads the data from all of the NII files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(self, dataset_config, phase):
        self.subdataset = dataset_config.get('subdataset', 'StandardNIIDataset')
        self.dataset_class = StandardNIIDataset
        if self.subdataset == 'CoarseSegmentedNIIDataset':
            self.dataset_class = CoarseSegmentedNIIDataset
        self.datasets = None
        self.dataset_config = dataset_config
        self.phase_config = dataset_config[phase]
        # load data augmentation configuration
        self.transformer_config = self.phase_config['transformer']
        # load slice builder config
        self.slice_builder_config = self.phase_config['slice_builder']
        # load files to process
        # group num, so we can compute how many samples each group have
        self.k = dataset_config.get('group_num', 5)
        self.patch_count = 0
        self.current_group = 0
        self.start_ids = [0] * self.k
        self.phase = phase
        self.raw_group = None
        self.label_group = None
        self.weight_group = None
        self.mask_group = None
        shuffle = self.dataset_config.get('shuffle', False)
        self.init_datasets(shuffle=shuffle)
    def __getitem__(self, idx):
        gid, id_in_group = self._get_group_id(idx)
        if self.current_group is not gid:
            # read the datasets
            self.datasets = self.load_datasets(gid)
            self.current_group = gid
        return self.datasets.__getitem__(id_in_group)
    ## get groups id and local id in groups
    def _get_group_id(self, idx):
        for gid, si in enumerate(self.start_ids[1:]):
            if idx < si:
                return gid, idx - self.start_ids[gid]
        return len(self.start_ids) - 1, idx - self.start_ids[-1]
    def __len__(self):
        return self.patch_count 
    def load_datasets(self, gid):
        # load patches directly from npy, instead of raw files
        tmp_datasets = []
        for raw_file_path, label_file_path, weight_file_path, mask_file_path in zip(self.raw_group[gid], self.label_group[gid], self.weight_group[gid], self.mask_group[gid]):
            try:
                logger.info(f'Loading {self.phase} set from: {raw_file_path}...')
                if self.dataset_class == StandardNIIDataset:
                    dataset = self.dataset_class(raw_file_path=raw_file_path,
                                label_file_path =label_file_path,
                                weight_file_path=weight_file_path,
                                phase=self.phase, 
                                slice_builder_config=self.slice_builder_config,
                                transformer_config=self.transformer_config,
                                mirror_padding=self.dataset_config.get('mirror_padding', None),
                                global_normalization=self.dataset_config.get('global_normalization', None))
                else:
                    dataset = self.dataset_class(raw_file_path=raw_file_path,
                                label_file_path =label_file_path,
                                weight_file_path=weight_file_path,
                                cmask_file_path = mask_file_path,
                                phase=self.phase, 
                                slice_builder_config=self.slice_builder_config,
                                transformer_config=self.transformer_config,
                                mirror_padding=self.dataset_config.get('mirror_padding', None),
                                global_normalization=self.dataset_config.get('global_normalization', None))                    
                tmp_datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {self.phase} set: {raw_file_path}', exc_info=True)            

        concat_dataset = ConcatDataset(tmp_datasets)
        return concat_dataset
    ### compute dataset patches, for each subdataset, save the slice indices
    def init_datasets(self, shuffle = False):

        file_paths = self.phase_config['file_paths']
        raw_dir = self.dataset_config.get('raw_dir', 'raw')
        label_dir = self.dataset_config.get('label_dir', None)
        weight_dir = self.dataset_config.get('weight_dir', None)
        mask_path = self.phase_config.get('mask_path', None)
        raw_file_paths = self.dataset_class.traverse_nii_paths(file_paths, raw_dir)
        if label_dir is not None:
            label_file_paths = self.dataset_class.traverse_nii_paths(file_paths, label_dir)
        else:
            label_file_paths = [None for _ in raw_file_paths]
        if weight_dir is not None:
            weight_file_paths = self.dataset_class.traverse_nii_paths(file_paths, weight_dir)
        else:
            weight_file_paths = [None for _ in raw_file_paths]
        if mask_path is not None:
            mask_file_paths = self.dataset_class.traverse_nii_paths([mask_path], '')
        else:
            mask_file_paths = [None for _ in raw_file_paths]
        if shuffle:
            rand_id = np.random.shuffle([_ for _ in range(len(raw_file_paths))])
            raw_file_paths = [ raw_file_paths[r] for r in rand_id]
            label_file_paths = [ label_file_paths[r] for r in rand_id]
            weight_file_paths = [ weight_file_paths[r] for r in rand_id]
            mask_file_paths = [ mask_file_paths[r] for r in rand_id]
        step = math.ceil(len(raw_file_paths) / self.k)
        path_len = len(raw_file_paths)
        self.raw_group = [raw_file_paths[i:i+step] for i in range(0, path_len, step) ]
        self.label_group = [label_file_paths[i:i+step] for i in range(0, path_len, step) ]
        self.weight_group = [weight_file_paths[i:i+step] for i in range(0, path_len, step) ]
        self.mask_group = [mask_file_paths[i:i+step] for i in range(0, path_len, step)]
        self.k = len(self.raw_group)
        self.patch_count = 0
        for gid in range(len(self.raw_group)):
            print(gid, self.patch_count)
            self.start_ids[gid] = self.patch_count
            concat_dataset = self.load_datasets(gid)
            self.patch_count += len(concat_dataset)
            if gid == 0:
                self.datasets = concat_dataset
            
    @classmethod
    def create_datasets(cls, dataset_config, phase):
        return cls(dataset_config, phase)