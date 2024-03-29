# this file is used to compute prediction error.
# Betti number error (0, 1)
# adapted rand index
# dice coefficient
# accuracy
# Variation of Information(VOI)
# Street Mover distance

from pyramid3dunet.np_metrics import MeanIoU, Dice, Accuracy, VariationOfInformation, AdaptedRandError, ConnectedComponentError
import torch, numpy as np
from collections.abc import Iterable
import sys, getopt
import os
import glob
from pyramid3dunet.utils import zoom, load_itk
from skimage.morphology import remove_small_objects
import scipy.ndimage as ndimage

def f(pred_path, label_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size=None, prob_threshold = 0.5, area_threshold = 50):
    label, _, _ = load_itk(label_path)
    label = label.astype(float)    

    pred, _, _ = load_itk(pred_path)
    pred = pred.astype(float)
    
    if zoom_size is not None:
        pred = zoom(pred, target_shape = zoom_size)

    if pred.shape != label.shape:
        # label = zoom(label, target_shape = pred.shape)
        # print(pred.shape, label.shape)
        pred = zoom(pred, target_shape = label.shape)


    label[ label >= 0.5 ] = 1.0
    label[ label < 0.5 ] = 0.0

    pred = remove_small_objects(pred > prob_threshold, min_size = area_threshold, connectivity = 3).astype(float)
    label = remove_small_objects(label > prob_threshold, min_size = area_threshold, connectivity = 3).astype(float)
    # label = label > prob_threshold
    ## for shape reconstruction from skeletal representation
    # e_label = ndimage.binary_erosion(label)
    # label = e_label.astype(float) +  label.astype(float) - ndimage.binary_dilation(e_label).astype(float)
    res = []
    for eval_c in eval_criterions_pixel:
        tmp = eval_c(pred, label)
        if torch.is_tensor(tmp):
            tmp = tmp.item()
        if isinstance(tmp, Iterable):
            res = res + list(tmp)
        else:
            res.append(tmp)

    structure = np.ones((3,3,3), int)
    # print(pred.shape, structure.shape)
    input_label, _ = ndimage.label(np.logical_not(pred), structure = structure)
    target_label, _ = ndimage.label(np.logical_not(label), structure = structure)  
        
    for eval_c in eval_criterions_cluster:
        tmp = eval_c(input_label, target_label, False)
        if torch.is_tensor(tmp):
            tmp = tmp.item()
        # print(tmp)
        if isinstance(tmp, Iterable):
            res = res + list(tmp)
        else:
            res.append(tmp)
    return res

def main():
    argv = sys.argv[1:]
    pred_path = ''
    gt_path = ''
    threshold = 0.5
    opts, _ = getopt.getopt(argv, "hp:g:t:z:a:", ['help', 'pred=', 'gt=', 'threshold=', 'zoom=', 'area='])
    zoom_size = None
    area = 50
    if len(opts) == 0:
        print('unknow options, usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -z <zoom = None> -a <area = 50>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5> -z <zoom = None> -a <area = 50>')
            sys.exit()
        elif opt in ("-p", '--pred'):
            pred_path = arg
        elif opt in ("-g", '--gt'):
            gt_path = arg
        elif opt in ("-t", '--threshold'):
            threshold = float(arg)
        elif opt in ('-z', '--zoom'):
            zoom_size = tuple([int(z) for z in arg.split(',')])
        elif opt in ('-a', '--area'):
            area = int(arg)
        else:
            print('unknow option,usage: evaluate.py -p <pred_file> -g <gt_file> -t <threshold = 0.5>, -z <zoom = None> -a <area = 50>')
            sys.exit()
    # # construct evaluation metrics
    acc = Accuracy(threshold = threshold)
    dice = Dice(threshold = threshold)
    mIoU = MeanIoU(threshold = threshold)
    vio = VariationOfInformation(threshold = threshold)
    betti = ConnectedComponentError(threshold = threshold, dim=3)
    are = AdaptedRandError(threshold = threshold)
    eval_criterions_pixel = [acc, dice, mIoU, betti]
    eval_criterions_cluster = [vio, are]
    
    if os.path.isdir(pred_path):
        pred_paths = sorted(glob.glob(os.path.join(pred_path, '*')))
        gt_paths = sorted(glob.glob(os.path.join(gt_path, '*')))
        res = []
        for pfile, gfile in zip(pred_paths, gt_paths):
            # print(pfile, gfile)
            tmp_res = f(pfile, gfile, eval_criterions_pixel, eval_criterions_cluster, zoom_size, area_threshold = area)
            # res.append(tmp_res)
            hasnan = True if True in np.isnan(np.array(tmp_res)) else False
            if not hasnan:
                res.append(tmp_res)
            # print(tmp_res)
        res = np.mean( np.array(res), axis = 0 )
    else:
        res = f(pred_path, gt_path, eval_criterions_pixel, eval_criterions_cluster, zoom_size, area_threshold = area)

    print('acc: {}\ndice: {}\nmIoU: {}\nbetti0: {}\nbetti1: {}\nvio: {}\nare: {}'.format(res[0], res[1], res[2], res[3], res[4], res[5], res[6]))
if __name__ == "__main__":
    main()