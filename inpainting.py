from unittest.mock import patch
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from random import sample, shuffle
import cv2

from pixel import Pixel, PixelMap
from priority import get_patch, compute_confidence, compute_isophotes, compute_data_term


scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                    [-10+0j, 0+ 0j, +10 +0j],
                    [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

def load_image(image_file):
    im = cv2.imread(image_file,
                    cv2.IMREAD_COLOR)
    return im

def load_mask(mask_file, threshold=100):
    mask = cv2.imread(mask_file,
                      cv2.IMREAD_GRAYSCALE)
    mask[mask >= threshold] = 255
    mask[mask < threshold] = 0
    return mask

def get_contour(mask):
    # ATTENTION: peut-être vérifier que tous les points du contours sont à 255 dans le masque
    contour = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0][0]
    contour = np.flip(contour.squeeze(), axis=1)
    return contour

def show_contour(mask, contour, idx_to_show=None):
    new_mask = np.repeat(mask[:,:,None], 3, axis=2)
    for ind in contour:
        new_mask[ind[0],ind[1]] = [255,0,0]
    plt.figure()
    plt.imshow(new_mask, interpolation="none")
    if idx_to_show is not None:
        for idx in idx_to_show:
            point = contour[idx]
            normal = get_normal(point[0], point[1], mask)
            factor = 1e1
            plt.plot([point[1], point[1] + normal[1]*factor],
                    [point[0], point[0] + normal[0]*factor],
                    linestyle='-', linewidth=1, color="green")
    plt.axis("off")
    plt.show()

def get_normal(x, y, mask):
    patch = mask[x-1:x+2,y-1:y+2]
    assert patch.size == 9
    grad = -np.sum(patch*scharr)
    gx = np.real(grad)
    gy = np.imag(grad)
    normal = np.array([gy, gx])
    norm = np.linalg.norm(normal)
    if norm == 0:
        return normal
    normal = normal / norm
    return normal

def inpainting(im, mask):
    pixel_map = PixelMap(im, mask)
    contour = get_contour(mask)
    for p in contour:
        pixel_map.udpate_confidence(p)
    return


if __name__ == "__main__":
    image_file = "images/image3.jpg"
    im = load_image(image_file)
    print("Image :", im.shape)

    mask_file = "images/mask3.jpg"
    mask = load_mask(mask_file)
    print("Masque :", mask.shape)

    pixel_map = get_pixel_map(im)
    print(len(pixel_map))
    print(len(pixel_map[0]))

    # cv2.imshow("Image", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    grad = convolve2d(mask, scharr)
    grady = np.real(grad)
    gradx = np.imag(grad)

    # plt.figure()
    # plt.imshow(gradx)
    # plt.show()
    # plt.figure()
    # plt.imshow(grady)
    # plt.show()
    # plt.figure()
    # plt.imshow(gradx**2 + grady**2)
    # plt.show()


    contour = get_contour(mask)
    idx_all = range(len(contour))
    idx_to_show = idx_all
    show_contour(mask, contour, idx_to_show)



    # cv2.imshow("Masked image", masked_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.figure()
    # plt.imshow(np.flip(masked_im, axis=2), interpolation="none")
    # plt.show()

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
