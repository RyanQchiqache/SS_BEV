import os.path
import glob
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Tuple
import numpy as np
from codeBase.data.DataPreprocessor import DataPreprocessor
import PIL.Image as I
class SatelliteDataset(Dataset):
    def __init__(
        self,
        images_paths: np.ndarray,
        masks_paths: np.ndarray,
        transform: Optional[Callable] = None,
        rgb_to_class : Optional[Callable] = None
    ):
        assert len(images_paths) == len(masks_paths)
        self.images = images_paths
        self.masks = masks_paths
        self.transform = transform
        self.rgb_to_class = rgb_to_class

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = np.array(I.open(self.images[idx]).convert("RGB"))
        mask_rgb = np.array(I.open(self.masks[idx]).convert("RGB"))

        if self.rgb_to_class:
            mask = self.rgb_to_class(mask_rgb)
        else:
            mask = mask_rgb

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask

    @staticmethod
    def collate_fn(batch):
        images, masks = zip(*batch)
        return torch.stack(images), torch.stack(masks)



class FlairDataset(SatelliteDataset):
    COLOR_MAP = {
        1: ['building', '#db0e9a'],
        2: ['pervious surface', '#938e7b'],
        3: ['impervious surface', '#f80c00'],
        4: ['bare soil', '#a97101'],
        5: ['water', '#1553ae'],
        6: ['coniferous', '#194a26'],
        7: ['deciduous', '#46e483'],
        8: ['brushwood', '#f3a60d'],
        9: ['vineyard', '#660082'],
        10: ['herbaceous vegetation', '#55ff00'],
        11: ['agricultural land', '#fff30d'],
        12: ['plowed land', '#e4df7c'],
        13: ['swimming_pool', '#3de6eb'],
        14: ['snow', '#ffffff'],
        15: ['clear cut', '#8ab3a0'],
        16: ['mixed', '#6b714f'],
        17: ['ligneous', '#c5dc42'],
        18: ['greenhouse', '#9999ff'],
        19: ['other', '#000000'],
    }

    COLOR_TO_CLASS = DataPreprocessor.create_color_to_class(COLOR_MAP)

    @staticmethod
    def rgb_to_class(mask_rgb: np.ndarray) -> np.ndarray:
        H, W, _ = mask_rgb.shape
        class_map = np.zeros((H, W), dtype=np.uint8)
        for color, class_idx in FlairDataset.COLOR_TO_CLASS.items():
            matches = np.all(mask_rgb == color, axis=-1)
            class_map[matches] = class_idx
        return class_map

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        super().__init__(image_paths, mask_paths, transform, rgb_to_class=FlairDataset.rgb_to_class)


class DLRDataset(SatelliteDataset):
    COLOR_MAP_dense = {
        1: ['Low vegetation', '#f423e8'],
        2: ['Paved road', '#66669c'],
        3: ['Non paved road', '#be9999'],
        4: ['Paved parking place', '#999999'],
        5: ['Non paved parking place', '#faaa1e'],
        6: ['Bikeways', '#98fb98'],
        7: ['Sidewalks', '#4682b4'],
        8: ['Entrance exit', '#6b8e23'],
        9: ['Danger area', '#dcdc00'],
        10: ['Lane-markings', '#ff0000'],
        11: ['Building', '#dc143c'],
        12: ['Car', '#7d008e'],
        13: ['Trailer', '#aac828'],
        14: ['Van', '#c83c64'],
        15: ['Truck', '#961250'],
        16: ['Long truck', '#51b451'],
        17: ['Bus', '#bef115'],
        18: ['Clutter', '#0b7720'],
        19: ['Impervious surface', '#78f078'],
        20: ['Tree', '#464646'],
    }
    COLOR_MAP_multi_lane = {
        0: ['Background', '#000000'],
        1: ['Dash Line', '#ff0000'],
        2: ['Long Line', '#0000ff'],
        3: ['Small dash line', '#ffff00'],
        4: ['Turn signs', '#00ff00'],
        5: ['Other signs', '#ff8000'],
        6: ['Plus sign on crossroads', '#800000'],
        7: ['Crosswalk', '#00ffff'],
        8: ['Stop line', '#008000'],
        9: ['Zebra zone', '#ff00ff'],
        10: ['No parking zone', '#009696'],
        11: ['Parking space', '#c8c800'],
        12: ['Other lane-markings', '#6400c8'],
    }
    COLOR_TO_CLASS_dense = DataPreprocessor.create_color_to_class(COLOR_MAP_dense)
    COLOR_TO_CLASS_multi_lane = DataPreprocessor.create_color_to_class(COLOR_MAP_multi_lane)

    @staticmethod
    def convert_rgb_to_class(mask_rgb: np.ndarray, color_to_class: Dict[Tuple[int, int, int], int]) -> np.ndarray:
        H, W, _ = mask_rgb.shape
        class_map = np.zeros((H, W), dtype=np.uint8)
        for color, class_idx in color_to_class.items():
            matches = np.all(mask_rgb == color, axis=-1)
            class_map[matches] = class_idx
        return class_map

    def __init__(self, image_dir: str, mask_dir: str, label_type: str, transform=None):
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        if label_type == "dense":
            color_to_class = self.COLOR_TO_CLASS_dense
        elif label_type == "multilane":
            color_to_class = self.COLOR_TO_CLASS_multi_lane
        else:
            raise ValueError(f"Unknown label type: {label_type}")

        super().__init__(
            image_paths, mask_paths, transform,
            rgb_to_class=lambda mask_rgb: DLRDataset.convert_rgb_to_class(mask_rgb, color_to_class)
        )









