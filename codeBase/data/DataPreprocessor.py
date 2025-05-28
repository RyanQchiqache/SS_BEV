import os
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Any


class DataPreprocessor:
    """
    Preprocess dataset images and masks:
    - Load files from unified folders
    - Patchify into 256x256 tiles
    - Normalize images
    - Convert RGB mask colors to class labels
    """
    COLOR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
        (60, 16, 152): 0,     # Building
        (132, 41, 246): 1,    # Land (unpaved)
        (110, 193, 228): 2,   # Road
        (254, 221, 58): 3,    # Vegetation
        (226, 169, 41): 4,    # Water
        (155, 155, 155): 5    # Unlabeled
    }

    def __init__(self, image_dir: str, mask_dir: str, patch_size: int):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

    def rgb_to_class(self, mask_rgb: np.ndarray) -> np.ndarray:
        H, W, _ = mask_rgb.shape
        class_map = np.zeros((H, W), dtype=np.uint8)
        for color, class_idx in self.COLOR_TO_CLASS.items():
            matches = np.all(mask_rgb == color, axis=-1)
            class_map[matches] = class_idx
        return class_map

    def _patchify_image(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List[Any], List[Any], List[Any]]:
        H, W = mask.shape[:2]
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        image = image[:new_H, :new_W]
        mask = mask[:new_H, :new_W]

        h_patches = new_H // self.patch_size
        w_patches = new_W // self.patch_size
        img_patches = []
        mask_patches = []
        coords = []

        for i in range(h_patches):
            for j in range(w_patches):
                y0 = i * self.patch_size
                x0 = j * self.patch_size
                img_patch = image[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
                mask_patch = mask[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)
                coords.append((y0, x0))

        return img_patches, mask_patches, coords

    def prepare_data(self, train_split: float = 0.8, debug_limit: int = None):
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        assert len(image_files) == len(mask_files), "Mismatch between images and masks"

        total_images = len(image_files)
        split_idx = int(train_split * total_images)

        train_imgs, train_masks = [], []
        val_imgs, val_masks = [], []

        for idx in range(split_idx):
            img = np.array(Image.open(os.path.join(self.image_dir, image_files[idx])).convert("RGB"))
            mask_rgb = np.array(Image.open(os.path.join(self.mask_dir, mask_files[idx])).convert("RGB"))
            mask = self.rgb_to_class(mask_rgb)

            train_imgs.append(img)
            train_masks.append(mask)

        for idx in range(split_idx, total_images):
            img = np.array(Image.open(os.path.join(self.image_dir, image_files[idx])).convert("RGB"))
            mask_rgb = np.array(Image.open(os.path.join(self.mask_dir, mask_files[idx])).convert("RGB"))
            mask = self.rgb_to_class(mask_rgb)

            val_imgs.append(img)
            val_masks.append(mask)

        if debug_limit is not None:
            train_imgs = train_imgs[:debug_limit]
            train_masks = train_masks[:debug_limit]
            val_imgs = val_imgs[:debug_limit]
            val_masks = val_masks[:debug_limit]

        return (train_imgs, train_masks, val_imgs, val_masks)

