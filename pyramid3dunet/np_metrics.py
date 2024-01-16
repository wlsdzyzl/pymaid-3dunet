import numpy as np

from skimage.metrics import variation_of_information as voi, adapted_rand_error as are
from sklearn.metrics.cluster import adjusted_rand_score as ari

import scipy.ndimage as ndimage


## Error of connected components
# full structure
class ConnectedComponentError:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target):


        input = (input > self.threshold).astype(int)
        target = (target > self.threshold).astype(int)
        _, nf_i = ndimage.label(input, structure = self.structure)
        _, nf_t = ndimage.label(target, structure = self.structure)
        b0_error = np.abs(0.0 + nf_i - nf_t)

        input = 1 - input
        target = 1 - target
        _, nf_i = ndimage.label(input, structure = self.structure)
        _, nf_t = ndimage.label(target, structure = self.structure)
        b1_error = np.abs(0.0 + nf_i - nf_t)
        return [b0_error, b1_error] 
class MeanIoU:
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
    def __call__(self, input, target):
        input = (input > self.threshold).astype(float)
        target = (target > self.threshold).astype(float)
        return (input * target).sum() / (input.sum() + target.sum() - (input * target).sum())
class Dice:
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
    def __call__(self, input, target):
        input = (input > self.threshold).astype(float)
        target = (target > self.threshold).astype(float)
        return 2 * (input * target).sum() / (input.sum() + target.sum()) 
class Accuracy:
    def __init__(self, threshold = 0.5, **kwargs):
        self.threshold = threshold
    def __call__(self, input, target):
        input = (input > self.threshold).astype(float)
        target = (target > self.threshold).astype(float)
        res = (input == target).astype(float)
        return np.sum(res) / np.size(res)

# street mover distance (wait to be implemented)
class VariationOfInformation:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target, is_boundary = True):
        if is_boundary:
            input = (input < self.threshold).astype(int)
            target = (target < self.threshold).astype(int)
            input_label, _ = ndimage.label(input, structure = self.structure)
            target_label, _ = ndimage.label(target, structure = self.structure)
            return sum(voi(input_label, target_label)) / 2
        else:
            return sum(voi(input, target)) / 2

class AdjustedRandIndex:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target, is_boundary = True):
        if is_boundary:
            input = (input < self.threshold).astype(int)
            target = (target < self.threshold).astype(int)
            input_label, _ = ndimage.label(input, structure = self.structure)
            target_label, _ = ndimage.label(target, structure = self.structure)
            return ari(input_label.flatten(), target_label.flatten())
        else:
            return ari(input.flatten(), target.flatten())
class AdaptedRandError:
    def __init__(self, threshold = 0.5, dim = 2, **kwargs):
        self.threshold = threshold
        self.structure = np.ones((3,3), int)
        if dim == 3:
            self.structure = np.ones((3,3,3), int)
    def __call__(self, input, target, is_boundary = True):
        if is_boundary:
            input = (input < self.threshold).astype(int)
            target = (target < self.threshold).astype(int)
            input_label, _ = ndimage.label(input, structure = self.structure)
            target_label, _ = ndimage.label(target, structure = self.structure)
            return are(input_label, target_label)[0]
        else:
            return are(input, target)[0]
