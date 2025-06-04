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
    - Reconstruct full images from patches
    """
    COLOR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
        (60, 16, 152): 0,     # Building
        (132, 41, 246): 1,    # Land (unpaved)
        (110, 193, 228): 2,   # Road
        (254, 221, 58): 3,    # Vegetation
        (226, 169, 41): 4,    # Water
        (155, 155, 155): 5    # Unlabeled
    }

    def __init__(self, image_dir: str, mask_dir: str, patch_size: int, overlap: int = 0):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.overlap = overlap

    def rgb_to_class(self, mask_rgb: np.ndarray) -> np.ndarray:
        H, W, _ = mask_rgb.shape
        class_map = np.zeros((H, W), dtype=np.uint8)
        for color, class_idx in self.COLOR_TO_CLASS.items():
            matches = np.all(mask_rgb == color, axis=-1)
            class_map[matches] = class_idx
        return class_map

    def _patchify_image_old(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List[Any], List[Any], List[Any]]:
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

    def patchify_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
        ps = self.patch_size
        step = ps - self.overlap
        H, W = image.shape[:2]

        pad_bottom = (ps - H % ps) % ps
        pad_right = (ps - W % ps) % ps

        if pad_bottom or pad_right:
            image = np.pad(
                image,
                ((0, pad_bottom), (0, pad_right), (0, 0)) if image.ndim == 3 else ((0, pad_bottom), (0, pad_right)),
                mode='constant',
                constant_values=0
            )
            H, W = image.shape[:2]

        patches = []
        coordinates = []

        for top in range(0, H - ps + 1, step):
            for left in range(0, W - ps + 1, step):
                patch = image[top:top + ps, left:left + ps]
                patches.append(patch)
                coordinates.append((top, left))

        return patches, coordinates, (H, W)

    def reconstruct_from_patches(self, patches: List[np.ndarray], coordinates: List[Tuple[int, int]],
                                 full_shape: Tuple[int, int]) -> np.ndarray:
        H, W = full_shape
        ps = self.patch_size
        is_rgb = patches[0].ndim == 3
        C = patches[0].shape[2] if is_rgb else 1

        canvas = np.zeros((H, W, C), dtype=np.float32) if is_rgb else np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for patch, (top, left) in zip(patches, coordinates):
            if is_rgb:
                canvas[top:top + ps, left:left + ps] += patch.astype(np.float32)
            else:
                canvas[top:top + ps, left:left + ps] += patch.astype(np.float32)
            weight[top:top + ps, left:left + ps] += 1.0

        weight[weight == 0] = 1.0
        if is_rgb:
            canvas = canvas / weight[..., None]
        else:
            canvas = canvas / weight

        return canvas.astype(patches[0].dtype)

    def prepare_data(self, train_split: float = 0.8, debug_limit: int = None):
        def load_image_and_mask(image_name, mask_name):
            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = np.array(Image.open(image_path).convert("RGB"))
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask = self.rgb_to_class(mask_rgb)

            return image, mask
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        assert len(image_files) == len(mask_files), "Mismatch between images and masks"

        combined = list(zip(image_files, mask_files))
        np.random.shuffle(combined)

        # split into training and validation sets
        total = len(combined)
        split_idx = int(train_split * total)
        train_pairs = combined[:split_idx]
        val_pairs = combined[split_idx:]

        # load image and mask data
        train_imgs, train_masks = zip(*[load_image_and_mask(img, mask) for img, mask in train_pairs])
        val_imgs, val_masks = zip(*[load_image_and_mask(img, mask) for img, mask in val_pairs])

        if debug_limit is not None:
            train_imgs = train_imgs[:debug_limit]
            train_masks = train_masks[:debug_limit]
            val_imgs = val_imgs[:debug_limit]
            val_masks = val_masks[:debug_limit]

        return list(train_imgs), list(train_masks), list(val_imgs), list(val_masks)

