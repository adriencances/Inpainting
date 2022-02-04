import numpy as np
from pixel import Pixel
import matplotlib.pyplot as plt
from scipy import signal

scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],

                   [-10+0j, 0+ 0j, +10 +0j],

                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy


def get_patch(p, Image, patch_size=9):
    """
    get a patch centered at pixel p

    Args:
        p (Pixel): center pixel
        Image (matrix of Pixel): matrix of pixel
        patch_size (int, optional): patch size. Defaults to 9.

    Returns:
        Matrix of Pixel: patch of pixels centered at p
    """
    half_patch_size = patch_size//2
    height, width = Image.shape
    min_x, max_x = max(0, p.x - half_patch_size), min(height, p.x+half_patch_size + 1)
    min_y, max_y = max(0, p.y - half_patch_size), min(width, p.y + half_patch_size + 1)
    return Image[min_x:max_x, min_y:max_y]


def compute_confidence(p, patch, patch_size=9):
    """compute confidence score of a pixel

    Args:
        p ([type]): [description]
        patch ([type]): [description]
        patch_size (int, optional): [description]. Defaults to 9.

    Returns:
        [type]: [description]
    """
    confidence = 0
    for row in patch:
        for p in row:
            confidence += p.confidence
    return confidence/patch.size


def compute_data_term(isophote, normal, alpha):
    return (isophote@normal)/alpha

def compute_isophote(p,Image, patch_size=9):
    """computing the isophote at a given pixel

    Args:
        p (Pixel): [description]
        Image (Matrix of pixels): [description]
        patch_size (int, optional): [description]. Defaults to 9.

    Returns:
        [type]: [description]
    """
    larger_patch = get_patch(p, Image, patch_size+2)
    larger_image = np.array([[float(p.value) for p in row] for row in larger_patch])
    grad = signal.convolve2d(larger_image, scharr, mode='valid')
    mask = np.where(larger_image[1:patch_size+1, 1:patch_size+1] >= 0, 1, 0) # computing the mask on the patch
    norm_grad = np.absolute(grad)
    norm_grad = mask*norm_grad
    i,j = np.unravel_index(norm_grad.argmax(), norm_grad.shape)
    return (-np.real(norm_grad[i,j]), np.imag(norm_grad[i,j]))   # returning -dy, dx
