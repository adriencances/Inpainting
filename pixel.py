from turtle import position
from unittest.mock import patch
import numpy as np
from scipy import signal
from dataclasses import dataclass

from utils import get_patch, rolling_window


@dataclass
class Pixel:
    value: int
    x: int
    y: int
    confidence: float = 0
    priority: float = 0
    data_term: float = 0

    def update_priority(self):
        self.priority = self.confidence*self.data_term


class PixelMap:
    def __init__(self, im, mask, patch_size=9, alpha=255):
        self.im = im.copy()
        self.im[mask > 0] = 0

        self.mask_init = mask.copy()
        self.mask = mask.copy()

        self.shape = im.shape[:2]
        self.height, self.width = self.shape

        self.patch_size = patch_size
        self.alpha = alpha

        self.pixel_map = [[Pixel(im[x,y], x, y) for y in range(self.width)] for x in range(self.height)]
        self.confidence = np.zeros(self.shape)
        self.confidence[mask == 0] = 1

        # VOIR SI C'EST REELEMENT UTILE
        self.gradx, self.grady = np.gradient(self.im)

        self.scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                                [-10+0j, 0+ 0j, +10 +0j],
                                [ -3+3j, 0+10j,  +3 +3j]])

    def __getitem__(self, x, y):
        return self.pixel_map[x][y]

    # VERIFIER SI SUBITLITE DANS ARTICLE
    def update_confidence(self, p):
        x, y = p
        confidence_patch = get_patch(p, self.confidence*np.logical_not(self.mask), self.patch_size)
        self.confidence[x,y] = confidence_patch.mean()

    def no_masked_neighbors_filter(self, mask, patch_size=3):
        kernel = np.ones((patch_size,patch_size))
        nb_of_masked_neighbors = signal.convolve2d(mask, kernel, mode='same')
        filter = (nb_of_masked_neighbors == 0)
        assert filter.shape == mask.shape
        return filter

    def compute_isophotes(self, contour):
        half_patch_size = self.patch_size//2
        isophotes = []
        filter = self.no_masked_neighbors_filter(self.mask)

        gradx, grady = np.gradient(self.im)
        filt_gradx = filter*gradx
        filt_grady = filter*grady
        module_grad = filt_gradx**2 + filt_grady**2

        for p in contour:
            x, y = p
            patch = get_patch(p, module_grad, self.patch_size)
            i,j = np.unravel_index(patch.argmax(), patch.shape)
            xb = x - half_patch_size + i
            yb = y - half_patch_size + j
            isophotes.append([-filt_grady[xb,yb], filt_gradx[xb,yb]])
        return isophotes

    def get_normal(self, p):
        patch = get_patch(p, self.mask, patch_size=3)
        grad = -np.sum(patch*self.scharr)
        gradx = np.real(grad)
        grady = np.imag(grad)

        normal = np.array([grady, gradx])
        norm = np.linalg.norm(normal)
        if norm == 0:
            return normal
        normal = normal / norm
        return normal

    def get_max_priority_pixel(self, contour):
        priorities = np.zeros(len(contour))
        isophotes = self.compute_isophotes(contour)
        for idx, p in enumerate(contour):
            x, y = p
            self.update_confidence(p)
            normal = self.get_normal(p)
            iso = isophotes[idx]
            data_term = abs(normal@iso)/self.alpha
            priorities[idx] = self.confidence[x,y] * data_term
        idx_max = priorities.argmax()
        p_hat = contour[idx_max]
        return p_hat.tolist()

    def get_best_patch(self, p_hat):
        filter = self.no_masked_neighbors_filter(self.mask_init, patch_size=self.patch_size)
        filter = filter[(self.patch_size//2):-(self.patch_size//2)]
        filter = filter[:,(self.patch_size//2):-(self.patch_size//2)]

        patch = get_patch(p_hat, self.im, self.patch_size)
        mask_patch = get_patch(p_hat, self.mask, self.patch_size)
        shape = (self.patch_size, self.patch_size)
        windows = rolling_window(self.im, shape)
        windows = windows*np.logical_not(mask_patch[None,None,:,:])

        ssds = np.sum((windows-patch[None,None,:,:])**2, axis=(2,3))
        assert ssds.shape == filter.shape
        ssds[np.logical_not(filter)] = np.inf

        idx_min = np.random.choice(np.flatnonzero(ssds == ssds.min()))
        q_hat = list(np.unravel_index(idx_min, ssds.shape))
        q_patch = windows[tuple(q_hat)]
        q_hat[0] += self.patch_size//2
        q_hat[1] += self.patch_size//2
        return q_hat, patch, q_patch

    def copy_image_data(self, p_hat, q_hat):
        assert isinstance(p_hat, list)
        assert isinstance(q_hat, list)

        xmin = max(p_hat[0] - (self.patch_size//2), 0)
        xmax = min(p_hat[0] + (self.patch_size//2) + 1, self.height)
        ymin = max(p_hat[1] - (self.patch_size//2), 0)
        ymax = min(p_hat[1] + (self.patch_size//2) + 1, self.width)
        X, Y = np.mgrid[xmin:xmax, ymin:ymax]
        positions = np.vstack([X.ravel(), Y.ravel()])

        nz_idx = np.nonzero(self.mask[tuple(positions)])[0]
        positions = positions[:,nz_idx]
        assert p_hat in positions.transpose(1,0).tolist()

        ref_positions = positions.copy()
        ref_positions[0] += q_hat[0] - p_hat[0]
        ref_positions[1] += q_hat[1] - p_hat[1]
        assert q_hat in ref_positions.transpose(1,0).tolist()

        self.im[tuple(positions)] = self.im[tuple(ref_positions)]
        self.mask[tuple(positions)] = False

        confidence_value = self.confidence[p_hat[0],p_hat[1]]
        for p in positions.transpose(1,0):
            x, y = p
            self.confidence[x,y] = confidence_value


