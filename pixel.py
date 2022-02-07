import numpy as np
from dataclasses import dataclass


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
    def __init__(self, im, mask):
        self.shape = im.shape[:2]
        self.height, self.width = self.shape
        self.pixel_map = [[Pixel(im[x,y], x, y) for y in range(self.width)] for x in range(self.height)]
        self.confidence = np.zeros(self.shape)
        self.confidence[mask == 0] = 1

    def __getitem__(self, x, y):
        return self.pixel_map[x][y]

    def update_confidence(self, p, half_patch_size=4):
        x, y = p
        xmin = max(x - half_patch_size, 0)
        xmax = min(x + half_patch_size + 1, self.height)
        ymin = max(y - half_patch_size, 0)
        ymax = min(y + half_patch_size + 1, self.width)
        confidence_patch = self.confidence[xmin:xmax, ymin:ymax]
        self.confidence[x,y] = confidence_patch.mean()

    def update_priority(self):
        self.priority = self.confidence*self.data_term
