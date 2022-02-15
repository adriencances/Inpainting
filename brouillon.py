from genericpath import exists
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from tqdm import tqdm
import argparse
from utils import get_patch, rolling_window, load_image, load_mask, rgb2gray, get_contour
from inpainting import *



parser = argparse.ArgumentParser()
parser.add_argument("--ps", dest="patch_size", type=int, default=9, help="patch size")
parser.add_argument("--it", dest="nb_iters", type=int, default=1, help="number of iterations")
parser.add_argument("--f", dest="image_and_mask_paths", nargs=2,
                    # default=["images/double_cross.jpg", "images/mask_double_cross.jpg"],
                    help="paths of image and mask")
parser.add_argument("--out", dest="output_dir", default=None, help="name of the output directory")

# Parse arguments
args = parser.parse_args()
image_file, mask_file = args.image_and_mask_paths
patch_size = args.patch_size
nb_iters = args.nb_iters

# Initialize pixel map object
im_rgb = load_image(image_file)
mask = load_mask(mask_file)
pixel_map = PixelMap(im_rgb, mask, patch_size=patch_size)
print("Image :", im_rgb.shape)
print("Masque :", mask.shape)
print(f"The patchsize is {pixel_map.patch_size}")

# Prepare output dir
output_dir = args.output_dir
if output_dir is None:
    output_dir = image_file.split("/")[-1].split(".")[0]
print(output_dir)
if output_dir is not None:
    Path(f"outputs/{output_dir}").mkdir(parents=True, exist_ok=True)


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

def show_contour(im, contour, p_hat, q_hat, p_patch, q_patch, mask, positions,
                 confidence, i, isophotes=None, idx_max=None, normals=None, priorities=None,
                 output_dir=None):
    cmap = cm.get_cmap("gray", 255)
    new_im = im.copy()
    # new_im = np.tile(im[:,:,None],3).astype(np.uint8)
    for p in contour:
        new_im[p[0], p[1]] = [255,0,0]
    new_im[mask] = [0,0,255]
    new_im[tuple(positions)] = [255,255,0]
    # new_im[p_hat[0]-4:p_hat[0]+5,p_hat[1]-4:p_hat[1]+5] = [0,0,255]

    gs = gridspec.GridSpec(2,4)
    gs.update(wspace=0.5)

    # fig, ax = plt.subplots(1,2)
    ax1 = plt.subplot(gs[0,:])
    ax2 = plt.subplot(gs[1,:2])
    ax3 = plt.subplot(gs[1,2:])

    ax1.imshow(new_im)
    ax1.scatter([p_hat[1]], [p_hat[0]], color="m", s=2)
    ax1.scatter([q_hat[1]], [q_hat[0]], color="m", s=2)

    ax2.imshow(p_patch, cmap=cmap, vmin = 0, vmax = 255)
    ax2.axis("off")
    ax3.imshow(q_patch, cmap=cmap, vmin = 0, vmax = 255)
    ax3.axis("off")
    # if not priorities is None:
    #     priority_map = np.zeros(im.shape)
    #     for p, priority in zip(contour, priorities):
    #         x1,y1= p
    #         priority_map[x1,y1] = priority
    #     imm = ax2.imshow(priority_map)
    # else:
    #     imm = ax2.imshow(confidence, interpolation='none')

    # if not isophotes is None:
    #     for iso, norm, p in zip(isophotes, normals, contour):
    #         # iso = isophotes[idx_max]
    #         x1, y1 = p
    #         if 85< y1 <110:
    #             ax[0].plot([y1, y1+iso[1]*0.1], [x1, x1 + iso[0]*0.1], color='red')
    #             ax[0].plot([y1, y1+norm[1]*10], [x1, x1+norm[0]*10], color='green' )
    # fig.colorbar(imm, ax=ax[1], fraction=0.046, pad=0.04)
    if output_dir is None:
        file_name = f"outputs/{i}"
    else:
        file_name = f"outputs/{output_dir}/{i}"
    plt.savefig(file_name)
    plt.close()


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


def get_ssds(pixel_map, p_hat):
    filter = pixel_map.no_masked_neighbors_filter(pixel_map.mask_init, patch_size=pixel_map.patch_size)
    filter = filter[(pixel_map.patch_size//2):-(pixel_map.patch_size//2)]
    filter = filter[:,(pixel_map.patch_size//2):-(pixel_map.patch_size//2)]

    patch = get_patch(p_hat, pixel_map.im_rgb, pixel_map.patch_size)
    # kernel = gaussian_filter(patch[:,:,0])
    # for c in range(3):
    #     patch[:,:,c] = signal.convolve2d(patch[:,:,c], kernel, mode="same")
    mask_patch = get_patch(p_hat, pixel_map.mask, pixel_map.patch_size)

    shape = (pixel_map.patch_size, pixel_map.patch_size, 3)
    windows = rolling_window(pixel_map.im_rgb, shape)
    assert windows.shape[2] == 1
    windows = windows*np.logical_not(mask_patch[None,None,None,:,:,None])

    ssds = np.sum((windows-patch[None,None,None,:,:,:])**2, axis=(2,3,4,5))
    assert ssds.shape == filter.shape
    return ssds

def show_ssds(pixel_map, p_hat, it):
    border = pixel_map.patch_size//2
    ssds = get_ssds(pixel_map, p_hat)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pixel_map.im_rgb_init)
    plt.scatter([p_hat[1]], [p_hat[0]], s=5, c="red")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(ssds)
    plt.scatter([p_hat[1] - border], [p_hat[0] - border], s=5, c="red")
    plt.axis("off")
    plt.savefig(f"outputs/ssd_{it}.png")


mask_sizes = []
mask_size = np.count_nonzero(pixel_map.mask)
mask_sizes.append(mask_size)
contour_length = []
log_file = "log.txt"
with open(log_file, "w") as f:
    f.write("Taille du patch\n")
for it in tqdm(range(nb_iters)):
    mask_size = np.count_nonzero(pixel_map.mask)
    mask_sizes.append(mask_size)
    if it%10 == 0:
        with open(log_file, "a") as f:
            f.write(f"{mask_size}\n")
    if mask_size == 0:
        break
    contour = get_contour(pixel_map.mask)
    contour_length.append(contour.shape[0])
    p_hat, isophotes, idx_max, normals, priorities = pixel_map.get_max_priority_pixel(contour)

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
        show_contour(pixel_map.im_rgb, contour, p_hat, q_hat, p_patch, q_patch,
                     pixel_map.mask, positions, pixel_map.confidence, it,
                     isophotes=isophotes, idx_max=idx_max, normals=normals,
                     priorities=priorities, output_dir=output_dir)
    # show_patches(p_patch, q_patch)
    if it > nb_iters - 5:
        show_ssds(pixel_map, p_hat, it)
    pixel_map.copy_image_data(p_hat, q_hat)



# plt.figure()
# plt.plot(np.arange(len(contour_length)), contour_length)
# plt.title('Nombre de points du contour')
# plt.xlabel('Numero iteration')
# plt.ylabel('Nb elements contour')
# plt.show()

