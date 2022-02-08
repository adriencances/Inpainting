from inpainting import *
import sys
from numpy import dtype
import matplotlib.pyplot as plt
from matplotlib import cm

from utils import get_patch, rolling_window, load_image, load_mask, get_contour

if len(sys.argv) > 1:
    nb_iters = int(sys.argv[1])
else:
    nb_iters = 1

def update_conf(pixel_map, update_mask=False):
    contour = get_contour(mask)
    if len(contour) == 0:
        return True
    for p in contour:
        x, y = p
        pixel_map.update_confidence(p)
        if update_mask:
            pixel_map.mask[x,y] = 0
    return False

def show(array, gray=False):
    if gray:
        plt.imshow(array, cmap="gray")
    else:
        plt.imshow(array)
    plt.axis("off")
    plt.show()

def show_contour(im, contour, p_hat, q_hat, p_patch, q_patch, mask, positions):
    cmap = cm.get_cmap("gray", 255)
    new_im = np.tile(im[:,:,None],3).astype(np.uint8)
    for p in contour:
        new_im[p[0], p[1]] = [255,0,0]
    new_im[mask] = [0,0,255]
    new_im[tuple(positions)] = [255,255,0]
    # new_im[p_hat[0]-4:p_hat[0]+5,p_hat[1]-4:p_hat[1]+5] = [0,0,255]
    plt.figure()
    plt.imshow(new_im)
    plt.scatter([p_hat[1]], [p_hat[0]], color="green", s=2)
    plt.scatter([q_hat[1]], [q_hat[0]], color="green", s=2)
    plt.show()

    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(new_im)
    # plt.scatter([p_hat[1]], [p_hat[0]], color="green", s=1)
    # plt.scatter([q_hat[1]], [q_hat[0]], color="green", s=1)
    # plt.subplot(2,2,3)
    # plt.imshow(p_patch, cmap=cmap, vmin = 0, vmax = 255)
    # plt.axis("off")
    # plt.subplot(2,2,4)
    # plt.imshow(q_patch, cmap=cmap, vmin = 0, vmax = 255)
    # plt.axis("off")
    # plt.show()

def show_result(im, mask):
    cmap = cm.get_cmap("gray", 255)
    new_im = np.tile(im[:,:,None],3).astype(np.uint8)
    new_im_masked = np.tile(im[:,:,None],3).astype(np.uint8)
    new_im_masked[mask] = [255,0,0]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(new_im_masked)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(new_im)
    plt.axis("off")
    plt.show()

def show_patches(p_patch, q_patch):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(p_patch)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(q_patch)
    plt.axis("off")
    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


image_file = "images/image3.jpg"
mask_file = "images/mask3.jpg"
im = load_image(image_file)
mask = load_mask(mask_file)
print("Image :", im.shape)
print("Masque :", mask.shape)


im = rgb2gray(im)
pixel_map = PixelMap(im, mask, patch_size=13)

mask_sizes = []
mask_size = np.count_nonzero(pixel_map.mask)
mask_sizes.append(mask_size)
for it in range(nb_iters):
    mask_size = np.count_nonzero(pixel_map.mask)
    mask_sizes.append(mask_size)
    if mask_size == 0:
        break
    contour = get_contour(pixel_map.mask)
    p_hat = pixel_map.get_max_priority_pixel(contour)
    q_hat, p_patch, q_patch = pixel_map.get_best_patch(p_hat)

    xmin = max(p_hat[0] - (pixel_map.patch_size//2), 0)
    xmax = min(p_hat[0] + (pixel_map.patch_size//2) + 1, pixel_map.height)
    ymin = max(p_hat[1] - (pixel_map.patch_size//2), 0)
    ymax = min(p_hat[1] + (pixel_map.patch_size//2) + 1, pixel_map.width)
    X, Y = np.mgrid[xmin:xmax, ymin:ymax]
    positions = np.vstack([X.ravel(), Y.ravel()])
    nz_idx = np.nonzero(pixel_map.mask[tuple(positions)])[0]
    positions = positions[:,nz_idx]

    if it >= 0:
        show_contour(pixel_map.im, contour, p_hat, q_hat, p_patch, q_patch, pixel_map.mask, positions)
    # show_patches(p_patch, q_patch)
    pixel_map.copy_image_data(p_hat, q_hat)

# show(pixel_map.mask)

# plt.figure()
# plt.plot(mask_sizes)
# plt.show()

show_result(pixel_map.im, mask)
# show(pixel_map.im, gray=True)



# # conf0 = pixel_map.confidence.copy()
# # for i in range(nb_iters):
# #     end = update_conf(pixel_map)
# #     if end:
# #         print(i)
# #         break
# # conf1 = pixel_map.confidence.copy()


# # plt.figure()
# # plt.subplot(1,2,1)
# # plt.imshow(conf0)
# # plt.subplot(1,2,2)
# # plt.imshow(conf1 - conf0)
# # plt.show()



