#!/bin/bash
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-unet/train.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-vnet/train.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-resunet/train.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-unet-sk/train.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_imagecas/5-fold-pyramidunet/train.sh

# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-pyramidunet-sk/train.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-respyramidunet/train.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-pyramidvnet/train.sh


# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-unet/predict.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-vnet/predict.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-resunet/predict.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-unet-sk/predict.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-pyramidunet/predict.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-pyramidunet-sk/predict.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_asoca/5-fold-pyramidunet/generate_mask.sh
# source /home/wlsdzyzl/project/pyramid-3dunet/resources/3DUnet_imagecas/5-fold-pyramidunet/generate_mask.sh

#### evaluate
echo '---------------------ASOCA----------------------'
echo '---------------------downsample------------------'
echo 'unet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/UNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo 'vnet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/VNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo 'resunet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/RESUNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo 'pyramidunet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/PyramidUNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo '---------------------origin------------------'
# echo 'unet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/UNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle/data/label
# echo 'vnet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/VNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle/data/label
# echo 'resunet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/RESUNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle/data/label
# echo 'pyramidunet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/PyramidUNET -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle/data/label
echo 'unet-patch':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/UNET-PATCH -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle/data/label
echo 'unet-stage-2':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_asoca/UNET-STAGE2 -g /media/wlsdzyzl/DATA1/datasets/ASOCA/nii/cropped_middle/data/label


echo '---------------------ImageCAS----------------------'
echo '---------------------downsample------------------'
echo 'unet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/UNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle_zoomed/data/label
echo 'vnet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/VNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle_zoomed/data/label
echo 'resunet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/RESUNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle_zoomed/data/label
echo 'pyramidunet':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/PyramidUNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle_zoomed/data/label
echo '---------------------origin------------------'
# echo 'unet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/UNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle/data/label
# echo 'vnet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/VNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle/data/label
# echo 'resunet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/RESUNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle/data/label
# echo 'pyramidunet':
# evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/PyramidUNET -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle/data/label
echo 'unet-patch':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/UNET-PATCH -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle/data/label
echo 'unet-stage-2':
evalpyramidunet -p /home/wlsdzyzl/project/pyramid-3dunet/generated_files/predictions/3d_imagecas/UNET-STAGE2 -g /media/wlsdzyzl/DATA1/datasets/imageCAS/sub_dataset_crop_middle/data/label
