import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_double_cross():
    height = 55
    width = 100
    im = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    im[:,15:30,0] = 255
    im[:,60:75,0] = 255
    im[20:35,:,2] = 255

    plt.imshow(im)
    plt.show()

generate_double_cross()
