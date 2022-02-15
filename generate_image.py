import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

def noisy(image):
    row,col,ch = image.shape
    mean = 0
    var = 100
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    print(gauss.min(), gauss.max())
    noisy = image + gauss[:,:,None]
    noisy[noisy < 0] = 0
    noisy[noisy > 255] = 255
    noisy = noisy.astype(np.uint8)
    return noisy


def generate_quadrillage():
    height = 160
    width = 160
    band_width = 20
    im = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            if (i-(height//2))**2 + (j-(width//2))**2 < 25**2:
                mask[i,j] = 255

    for k in range(height//(2*band_width)):
        im[2*k*band_width:(2*k+1)*band_width] = [255, 255, 255]

    # im = gaussian_filter(im, sigma=5)
    im = noisy(im)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(im)
    # plt.subplot(1,2,2)
    # plt.imshow(mask)
    # plt.show()

    im = Image.fromarray(im)
    im.save("images/noised_big_bandes.png")

    mask = Image.fromarray(mask)
    mask.save("images/mask_noised_big_bandes.png")



generate_quadrillage()
