import numpy as np


def get_patch(p, array, patch_size=9):
    x, y = p
    height, width = array.shape
    half_patch_size = patch_size//2
    xmin = max(x - half_patch_size, 0)
    xmax = min(x + half_patch_size + 1, height)
    ymin = max(y - half_patch_size, 0)
    ymax = min(y + half_patch_size + 1, width)
    return array[xmin:xmax, ymin:ymax]


def rolling_window(array, shape):  # rolling window for 2D array
    return np.lib.stride_tricks.sliding_window_view(array, shape)
