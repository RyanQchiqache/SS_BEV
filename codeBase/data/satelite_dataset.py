import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
import numpy as np

class SatelliteDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        transform: Optional[Callable] = None
    ):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

    @staticmethod
    def collate_fn(batch):
        images, masks = zip(*batch)
        return torch.stack(images), torch.stack(masks)
