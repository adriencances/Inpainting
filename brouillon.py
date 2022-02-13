from inpainting import *
import sys
from numpy import dtype
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from tqdm import tqdm
from utils import get_patch, rolling_window, load_image, load_mask, get_contour


im_number = 8

if len(sys.argv) > 1:
    nb_iters = int(sys.argv[1])
else:
    nb_iters = 1

if len(sys.argv) > 2:
    patch_size = int(sys.argv[2])
else:
    patch_size = 9

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

def show_contour(im, contour, p_hat, q_hat, p_patch, q_patch, mask, positions, confidence,i, isophotes=None, idx_max=None, normals=None, priorities=None):
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
    plt.savefig(f'outputs/{im_number}_{i}')
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


# image_file = f"images/image{im_number}.png"
image_file = f"images/pigeon.png"
mask_file = f"images/mask_pigeon.png"
# mask_file = f"images/mask{im_number}.png"
im_rgb = load_image(image_file)
im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
mask = load_mask(mask_file)
print("Image :", im_rgb.shape)
print("Masque :", mask.shape)


im = rgb2gray(im_rgb)
pixel_map = PixelMap(im, im_rgb, mask, patch_size=patch_size)
print(f'The patchsize is {pixel_map.patch_size}')
mask_sizes = []
mask_size = np.count_nonzero(pixel_map.mask)
mask_sizes.append(mask_size)
contour_length = []
for it in tqdm(range(nb_iters)):
    mask_size = np.count_nonzero(pixel_map.mask)
    mask_sizes.append(mask_size)
    if not it%10:
        print(mask_size)
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
        show_contour(pixel_map.im_rgb, contour, p_hat, q_hat, p_patch, q_patch, pixel_map.mask, positions, pixel_map.confidence, it, isophotes, idx_max, normals, priorities)
    # show_patches(p_patch, q_patch)
    pixel_map.copy_image_data(p_hat, q_hat)


# plt.figure()
# plt.plot(np.arange(len(contour_length)), contour_length)
# plt.title('Nombre de points du contour')
# plt.xlabel('Numero iteration')
# plt.ylabel('Nb elements contour')
# plt.show()


def get_ssds(pixel_map, p_hat):
    filter = pixel_map.no_masked_neighbors_filter(pixel_map.mask_init, patch_size=pixel_map.patch_size)
    filter = filter[(pixel_map.patch_size//2):-(pixel_map.patch_size//2)]
    filter = filter[:,(pixel_map.patch_size//2):-(pixel_map.patch_size//2)]

    patch = get_patch(p_hat, pixel_map.im_rgb, pixel_map.patch_size)
    mask_patch = get_patch(p_hat, pixel_map.mask, pixel_map.patch_size)
    shape = (pixel_map.patch_size, pixel_map.patch_size)
    windows = rolling_window(pixel_map.im_rg, shape)
    windows = windows*np.logical_not(mask_patch[None,None,None,:,:,None])

    ssds = np.sum((windows-patch[None,None,None,:,:,:])**2, axis=(2,3,4,5))
    assert ssds.shape == filter.shape
    # ssds[np.logical_not(filter)] = np.inf
    return ssds

def show_ssds(pixel_map, p_hat):
    ssds = get_ssds(pixel_map, p_hat)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pixel_map.im_init)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(ssds)
    plt.axis("off")
    plt.show()