import os
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List, Any
import cv2
import csv


class DataPreprocessor:
    """
    Preprocess dataset images and masks:
    - Load files from unified folders
    - Patchify into small tiles
    - Normalize images
    - Convert RGB mask colors to class labels
    - Reconstruct full images from patches
    """

    def __init__(self, image_dir: str, mask_dir: str, patch_size: int, overlap: int = 0, label_dict=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.overlap = overlap
        self.color_to_class = self.create_color_to_class(label_dict)

    def rgb_to_class(self, rgb_mask: np.ndarray) -> np.ndarray:
        class_map = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
        for color, class_idx in self.color_to_class.items():
            class_map[np.all(rgb_mask == color, axis=-1)] = class_idx
        return class_map


    def patchify_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]], Tuple[int, int]]:
        patch_size = self.patch_size
        step = patch_size - self.overlap
        H, W  = image.shape[:2]

        pad_bottom = (patch_size - H % patch_size) % patch_size
        pad_right = (patch_size - W % patch_size) % patch_size

        if pad_bottom or pad_right:
            image = np.pad( image, ((0, pad_bottom), (0, pad_right), (0, 0)) if image.ndim == 3 else ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
            H, W = image.shape[:2]

        patches = []
        coordinates = []

        for top in range(0, H - patch_size+ 1, step):
            for left in range(0, W - patch_size + 1, step):
                patch = image[top:top + patch_size, left:left + patch_size]
                patches.append(patch)
                coordinates.append((top, left))

        return patches, coordinates, (H, W)

    def patchify_dataset(self, images: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        img_patches: List[np.ndarray] = []
        mask_patches: List[np.ndarray] = []
        for img, mask in zip(images, masks):
            img_p, _, _ = self.patchify_image(img)
            mask_p, _, _ = self.patchify_image(mask)
            img_patches.extend(img_p)
            mask_patches.extend(mask_p)
        return img_patches, mask_patches

    @classmethod
    def reconstruct_from_patches(patches: List[np.ndarray], coordinates: List[Tuple[int, int]],
                                 full_shape: Tuple[int, int], patch_size) -> np.ndarray:
        H, W = full_shape
        ps = patch_size
        is_rgb = patches[0].ndim == 3
        C = patches[0].shape[2] if is_rgb else 1

        canvas = np.zeros((H, W, C), dtype=patches[0].dtype) if is_rgb else np.zeros((H, W), dtype=patches[0].dtype)
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

    def prepare_data(
            self,
            rgb_to_class,
            train_split: float = 0.8,
            debug_limit: int = None,
            use_csv: bool = False,
            train_csv_path: str = None,
            val_csv_path: str = None,
            base_dir: str = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        if use_csv:
            assert train_csv_path and val_csv_path, "CSV paths must be provided when use_csv=True"
            return self.prepare_data_from_csvs(train_csv_path, val_csv_path, rgb_to_class,base_dir, debug_limit)

        # Default: load from folders
        def load_image_and_mask(image_name, mask_name):
            image_path = os.path.join(self.image_dir, image_name)
            mask_path = os.path.join(self.mask_dir, mask_name)

            image = np.array(Image.open(image_path).convert("RGB"))
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask = self.rgb_to_class(mask_rgb)

            return image, mask

        image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith(('.png', '.jpg', '.tif'))])

        assert len(image_files) == len(mask_files), "Mismatch between images and masks"
        combined = list(zip(image_files, mask_files))
        np.random.shuffle(combined)

        split_idx = int(train_split * len(combined))
        train_pairs = combined[:split_idx]
        val_pairs = combined[split_idx:]

        # load image and mask data
        train_imgs, train_masks = zip(*[load_image_and_mask(img, mask) for img, mask in train_pairs])
        val_imgs, val_masks = zip(*[load_image_and_mask(img, mask) for img, mask in val_pairs])

        print(f"[Preprocessing] Loaded {len(train_imgs)} training and {len(val_imgs)} validation samples from folders.")

        if debug_limit is not None:
            train_imgs = train_imgs[:debug_limit]
            train_masks = train_masks[:debug_limit]
            val_imgs = val_imgs[:debug_limit]
            val_masks = val_masks[:debug_limit]

        return list(train_imgs), list(train_masks), list(val_imgs), list(val_masks)

    def prepare_data_from_csvs(
            self,
            train_csv_path: str,
            val_csv_path: str,
            rgb_to_class,
            base_dir=None,
            debug_limit: int = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        def load_csv(csv_path: str):
            with open(csv_path, newline='') as f:
                reader = csv.reader(f, delimiter=',')
                lines = [row for row in reader if len(row) == 2]
            return lines

        def resolve_path(p):
            return os.path.normpath(os.path.join(base_dir, p)) if base_dir and not os.path.isabs(p) else p

        def load_pair(image_path, mask_path):
            image_path = os.path.normpath(resolve_path(image_path))
            mask_path = os.path.normpath(resolve_path(mask_path))
            print(f"Image path: {image_path}")
            print(f"Mask path: {mask_path}")
            image = np.array(Image.open(image_path).convert("RGB"))
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask = self.rgb_to_class(mask_rgb)
            return image, mask
        train_lines = load_csv(train_csv_path)
        val_lines = load_csv(val_csv_path)

        train_data = [load_pair(*line) for line in train_lines]
        val_data = [load_pair(*line) for line in val_lines]

        train_imgs, train_masks = zip(*train_data)
        val_imgs, val_masks = zip(*val_data)

        if debug_limit is not None:
            train_imgs = train_imgs[:debug_limit]
            train_masks = train_masks[:debug_limit]
            val_imgs = val_imgs[:debug_limit]
            val_masks = val_masks[:debug_limit]

        print(f"[Preprocessing] Loaded {len(train_imgs)} training and {len(val_imgs)} validation samples from CSVs.")
        return list(train_imgs), list(train_masks), list(val_imgs), list(val_masks)

    @staticmethod
    def create_color_to_class(label_dict: dict) -> Dict[Tuple[int, int, int], int]:
        """
        Convert a dictionary with class indices and hex colors to a COLOR_TO_CLASS mapping.
        """
        if label_dict is None:
            raise ValueError("label_dict must be provided to create_color_to_class")

        def hex_to_rgb(hex_color: str) -> Tuple[int, ...]:
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        color_to_class = {}
        for class_index, (_, hex_color) in label_dict.items():
            rgb = hex_to_rgb(hex_color)
            color_to_class[rgb] = class_index

        return color_to_class



