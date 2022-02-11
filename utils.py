import numpy as np
import cv2
from scipy import signal

def get_patch(p, array, patch_size=9):
    x, y = p
    height, width = array.shape[0], array.shape[1]
    half_patch_size = patch_size//2
    xmin = max(x - half_patch_size, 0)
    xmax = min(x + half_patch_size + 1, height)
    ymin = max(y - half_patch_size, 0)
    ymax = min(y + half_patch_size + 1, width)
    return array[xmin:xmax, ymin:ymax]


def rolling_window(array, shape):  # rolling window for 2D array
    return np.lib.stride_tricks.sliding_window_view(array, shape)


def load_image(image_file):
    im = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return im


def load_mask(mask_file, threshold=100):
    mask = cv2.imread(mask_file,
                      cv2.IMREAD_GRAYSCALE)
    mask[mask >= threshold] = 255
    mask[mask < threshold] = 0
    return mask.astype(bool)


# def get_contour(mask):
#     # ATTENTION: peut-être vérifier que tous les points du contours sont à 255 dans le masque
#     contour = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
#     if len(contour) == 0:
#         return np.empty((0,0))
#     contour = contour[0]
#     contour = np.flip(contour.squeeze(axis=1), axis=1)
#     return contour

def get_contour(mask):
    kernel = np.ones((3,3))
    nb_neighbors = signal.convolve2d(mask, kernel, mode='same')
    filter = (nb_neighbors < 9)
    contour = np.transpose(np.nonzero(mask*filter))
    return contour


