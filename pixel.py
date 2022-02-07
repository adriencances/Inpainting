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
        self.im = im
        self.mask_init = mask
        self.mask = mask
        self.shape = im.shape[:2]
        self.height, self.width = self.shape

        self.patch_size = patch_size
        self.alpha = alpha

        self.pixel_map = [[Pixel(im[x,y], x, y) for y in range(self.width)] for x in range(self.height)]
        self.confidence = np.zeros(self.shape)
        self.confidence[mask == 0] = 1

        self.gradx, self.grady = np.gradient(im)

        self.scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                                [-10+0j, 0+ 0j, +10 +0j],
                                [ -3+3j, 0+10j,  +3 +3j]])

    def __getitem__(self, x, y):
        return self.pixel_map[x][y]

    def update_confidence(self, p):
        x, y = p
        confidence_patch = get_patch(p, self.confidence, self.patch_size)
        self.confidence[x,y] = confidence_patch.mean()

    def compute_isophotes(self, contour):
        half_patch_size = self.patch_size//2
        isophotes = []
        kernel = np.ones((3,3))/9
        nb_neighbors_in_mask = signal.convolve2d(self.mask, kernel, mode='same', boundary='symmetric')
        filter = (nb_neighbors_in_mask == 0)
        assert filter.shape == self.mask.shape

        filt_gradx = filter*self.gradx
        filt_grady = filter*self.grady
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
            normal = self.get_normal(p)
            iso = isophotes[idx]
            data_term = abs(normal@iso)/self.alpha
            priorities[idx] = self.confidence[x,y] * data_term
        idx_max = priorities.argmax()
        p_max = contour[idx_max]
        return p_max.tolist()

    def get_best_patch(self, p):
        patch = get_patch(p, self.im, self.patch_size)
        shape = (self.patch_size, self.patch_size)
        windows = rolling_window(self.im, shape)
        ssds = np.sum((windows-patch[None,None])**2, axis=(2,3))
        p_tild = np.unravel_index(ssds.argmin(), ssds.shape)
        return p_tild
