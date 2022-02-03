import numpy as np
import cv2
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    image_file = "images/image3.jpg"
    im = load_image(image_file)
    print(im.shape)

    mask_file = "images/mask3.jpg"
    mask = load_mask(mask_file)
    print(mask.shape)

    # cv2.imshow("Image", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    masked_im = im.copy() #p.zeros(im.shape)
    masked_im[mask > 0] = 0

    print(mask.min())
    print(mask.max())
    print(sorted(list(set(mask.flat))))

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0][0]
    contours = contours.squeeze()

    contour_colors = [mask[ind[1],ind[0]] for ind in contours]
    print("Colors of contour:", list(set(contour_colors)))

    # cv2.imshow("Masked image", masked_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    mask = np.repeat(mask[:,:,None], 3, axis=2)
    for ind in contours:
        mask[ind[1],ind[0]] = [0,255,0]
    plt.figure()
    plt.imshow(mask, interpolation="none", cmap="gray")
    plt.axis("off")
    plt.show()

    # plt.figure()
    # plt.imshow(np.flip(masked_im, axis=2), interpolation="none")
    # plt.show()

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
