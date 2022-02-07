#%%
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

# %%
