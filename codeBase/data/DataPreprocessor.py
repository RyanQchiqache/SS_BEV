import os
import cv2
import numpy as np
from typing import Dict, Tuple


class DataPreprocessor:
    """
    Preprocess dataset images and masks:
    - Load files from unified folders
    - Patchify into 256x256 tiles
    - Normalize images
    - Convert RGB mask colors to class labels
    """
    # Updated RGB color to class index mapping for the 6 classes
    COLOR_TO_CLASS: Dict[Tuple[int, int, int], int] = {
        (60, 16, 152): 0,  # Building
        (132, 41, 246): 1,  # Land (unpaved)
        (110, 193, 228): 2,  # Road
        (254, 221, 58): 3,  # Vegetation
        (226, 169, 41): 4,  # Water
        (155, 155, 155): 5  # Unlabeled
    }

    def __init__(self, image_dir: str, mask_dir: str, patch_size: int = 256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

    def rgb_to_class(self, mask_rgb: np.ndarray) -> np.ndarray:
        """
        Convert an RGB mask image (H×W×3) to a 2D array of class indices.
        """
        H, W, _ = mask_rgb.shape
        class_map = np.zeros((H, W), dtype=np.uint8)
        # Map each RGB color to the corresponding class index
        for color, class_idx in self.COLOR_TO_CLASS.items():
            matches = np.all(mask_rgb == color, axis=-1)
            class_map[matches] = class_idx
        return class_map

    def _patchify_image(self, image: np.ndarray, mask: np.ndarray):
        """
        Crop image & mask to the nearest multiple of patch_size, then split into patches.
        """
        H, W = mask.shape[:2]
        # Calculate nearest dimensions divisible by patch_size
        new_H = (H // self.patch_size) * self.patch_size
        new_W = (W // self.patch_size) * self.patch_size
        image = image[:new_H, :new_W]
        mask = mask[:new_H, :new_W]

        # Number of patches along each dimension
        h_patches = new_H // self.patch_size
        w_patches = new_W // self.patch_size
        img_patches = []
        mask_patches = []

        # Extract patches
        for i in range(h_patches):
            for j in range(w_patches):
                y0 = i * self.patch_size
                x0 = j * self.patch_size
                img_patch = image[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
                mask_patch = mask[y0:y0 + self.patch_size, x0:x0 + self.patch_size]
                img_patches.append(img_patch)
                mask_patches.append(mask_patch)
        return img_patches, mask_patches

    def prepare_data(self, train_split: float = 0.8):
        """
        Load all images and masks, patchify them, and split into training/validation sets.
        """
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))
        all_images = []
        all_masks = []

        for img_file, mask_file in zip(image_files, mask_files):
            # Construct the full paths
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            # Read image and mask using OpenCV (BGR) and convert to RGB
            bgr_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr_image is None:
                print(f"❌ Failed to load image: {img_path}")
                continue
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            bgr_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if bgr_mask is None:
                print(f"❌ Failed to load mask: {mask_path}")
                continue
            rgb_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2RGB)

            # Convert mask RGB to class indices
            class_mask = self.rgb_to_class(rgb_mask)

            # Patchify images and masks
            img_patches, mask_patches = self._patchify_image(rgb_image, class_mask)
            all_images.extend(img_patches)
            all_masks.extend(mask_patches)

        # Convert lists to numpy arrays
        all_images = np.array(all_images, dtype=np.uint8)
        all_masks = np.array(all_masks, dtype=np.uint8)

        # Split into training and validation sets
        total = len(all_images)
        train_count = int(total * train_split)
        indices = np.random.permutation(total)
        train_idx = indices[:train_count]
        val_idx = indices[train_count:]

        train_images = all_images[train_idx]
        train_masks = all_masks[train_idx]
        val_images = all_images[val_idx]
        val_masks = all_masks[val_idx]

        print(f"✅ Prepared data with {len(train_images)} training and {len(val_images)} validation samples.")
        return train_images, train_masks, val_images, val_masks
