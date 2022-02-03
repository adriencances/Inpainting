import numpy as np
from pixel import Pixel
import matplotlib.pyplot as plt

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
