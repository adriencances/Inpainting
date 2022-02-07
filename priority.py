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
    return abs(isophote@normal)/alpha

# def compute_isophote(p,Image, patch_size=9):
#     """computing the isophote at a given pixel

#     Args:
#         p (Pixel): [description]
#         Image (Matrix of pixels): [description]
#         patch_size (int, optional): [description]. Defaults to 9.

#     Returns:
#         [type]: [description]
#     """
#     larger_patch = get_patch(p, Image, patch_size+2)
#     larger_image = np.array([[float(p.value) for p in row] for row in larger_patch])
#     grad = signal.convolve2d(larger_image, scharr, mode='valid')
#     mask = np.where(larger_image[1:patch_size+1, 1:patch_size+1] >= 0, 1, 0) # computing the mask on the patch
#     norm_grad_module = np.absolute(grad)
#     norm_grad_module = mask*norm_grad_module
#     i,j = np.unravel_index(norm_grad_module.argmax(), norm_grad_module.shape)
#     return (-np.real(grad[i,j])/norm_grad_module[i,j], np.imag(grad[i,j])/norm_grad_module[i,j])   # returning -dy, dx


def compute_isophotes(mask, Image, contour_points, patch_size=9):
    h_patch_size = patch_size//2
    isophotes = []
    array_image = np.array([[p.value for p in row] for row in Image])
    kernel = np.ones((3,3))/9
    filter = signal.convolve2d(mask, kernel, mode='same', boundary='symmetric')
    filter = filter == 0

    gradx, grady = np.gradient(array_image)
    gradx = filter*gradx
    grady = filter*grady
    module_grad = gradx**2 + grady**2

    for p in contour_points:
        patch = module_grad[p[0]-h_patch_size: p[0]+h_patch_size+1, p[1]-h_patch_size:p[1]+h_patch_size+1]
        i,j = np.unravel_index(patch.argmax(), patch.shape)
        x = p[0] - h_patch_size + i
        y = p[1] - h_patch_size + j
        isophotes.append([-grady[x,y], gradx[x,y]])
    return isophotes
