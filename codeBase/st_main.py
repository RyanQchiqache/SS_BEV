import os
import torch
import numpy as np
import albumentations as A
from typing import List, Tuple,Any
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from codeBase.config.logging_setup import setup_logger, load_config
from codeBase.data.DataPreprocessor import DataPreprocessor
from codeBase.models.mask2former_model import Mask2FormerModel
from codeBase.visualisation.visualizer import Visualizer
from codeBase.data.satelite_dataset import SatelliteDataset, FlairDataset, DLRDataset
from datetime import datetime
from accelerate import Accelerator

DATASET_REGISTRY = {
    "satellite": SatelliteDataset,
    "flair": FlairDataset,
    "dlr": DLRDataset
}

class SegmentationPipeline:
    """
    A complete pipeline for semantic segmentation using Mask2Former on aerial imagery.
    Handles data preprocessing, augmentation, training, evaluation, and visualization.
    """
    def __init__(self) -> None:
        """Initializes the pipeline by loading configuration and setting up paths, logging, and randomness."""
        self.config = load_config()

        torch.manual_seed(42)
        np.random.seed(42)

        self.dataset_name = self.config["data"]["dataset_name"]
        self.label_type = self.config["data"].get("label_type", "dense")

        self.class_names = self.config["model"]["classes_names"].get(self.dataset_name, {}).get(self.label_type)
        if self.class_names is None:
            raise ValueError(
                f"Class names not found in config for dataset '{self.dataset_name}' with label type '{self.label_type}'")

        self.DatasetClass = DATASET_REGISTRY.get(self.dataset_name, SatelliteDataset)

        self.image_dir: str = self.config["data"]["images_dir"]
        self.mask_dir: str = self.config["data"]["masks_dir"]
        self.patch_size: int = int(self.config["data"]["patch_size"])
        self.batch_size: int = int(self.config["data"]["batch_size"])
        self.pretrained_weights: str = self.config["model"]["pretrained_weights"]
        self.num_classes: int = int(self.config["model"]["num_classes"])
        self.epochs: int = int(self.config["model"]["epochs"])
        self.learning_rate: float = float(self.config["model"]["learning_rate"])
        self.num_workers: int = int(self.config["model"]["num_workers"])
        self.pin_memory: bool = bool(self.config["model"]["pin_memory"])


        base_output_dir = self.config["paths"]["base_output_dir"]
        run_name = self.config["paths"].get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run_dir = os.path.join(base_output_dir, run_name)

        self.model_save_dir = os.path.join(self.run_dir, self.config["paths"]["checkpoint_subdir"])
        self.visualization_dir = os.path.join(self.run_dir, self.config["paths"]["visualization_subdir"])
        self.logs_dir = os.path.join(self.run_dir, self.config["paths"]["logs_subdir"])
        self.tensorboard_dir = os.path.join(self.run_dir, self.config["paths"]["tensorboard_subdir"])

        #self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(mixed_precision="fp16" if self.config["training"]["amp"] else "no")
        self.logger = setup_logger(__name__)
        self.device = self.accelerator.device
        self.logger.info(f"Using device: {self.device}")

        for dir_path in [self.run_dir, self.model_save_dir, self.visualization_dir, self.logs_dir,
                         self.tensorboard_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.logger = setup_logger(__name__)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
    def build_augmentation_pipeline(self) -> A.Compose:
        """
        Builds the Albumentations data augmentation pipeline based on configuration.

        Returns:
            A.Compose: Albumentations augmentation pipeline
        """
        aug_cfg = self.config["augmentation"]
        transforms_list = []

        if aug_cfg.get("horizontal_flip", 0) > 0:
            transforms_list.append(A.HorizontalFlip(p=aug_cfg["horizontal_flip"]))
        if aug_cfg.get("vertical_flip", 0) > 0:
            transforms_list.append(A.VerticalFlip(p=aug_cfg["vertical_flip"]))
        if aug_cfg.get("rotate_90", 0) > 0:
            transforms_list.append(A.RandomRotate90(p=aug_cfg["rotate_90"]))
        if aug_cfg.get("brightness_contrast", 0) > 0:
            transforms_list.append(A.RandomBrightnessContrast(p=aug_cfg["brightness_contrast"]))
        if aug_cfg.get("gaussian_blur", 0) > 0:
            transforms_list.append(A.GaussianBlur(p=aug_cfg["gaussian_blur"]))

        if aug_cfg.get("resized_crop", {}).get("enable", False):
            crop = aug_cfg["resized_crop"]
            transforms_list.append(
                A.RandomResizedCrop(
                    height=crop["height"],
                    width=crop["width"],
                    scale=tuple(crop["scale"]),
                    p=crop["p"]
                )
            )
        transforms_list.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        return A.Compose(transforms_list, additional_targets={"mask": "mask"})

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, List[np.ndarray], List[np.ndarray], DataPreprocessor]:
        """
        Loads and patchifies data, applies augmentations if configured, and prepares PyTorch DataLoaders.

        Returns:
            Tuple containing:
                - train_loader: DataLoader for training patches
                - val_loader: DataLoader for validation patches
                - val_imgs: Original validation images (for visualization)
                - val_masks: Original validation masks (for visualization)
                - preprocessor: The DataPreprocessor instance used
        """
        self.logger.info("Preparing data...")

        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            patch_size=self.patch_size,
            overlap=0
        )

        debug = self.config["data"].get("debug", False)
        debug_limit = self.config["data"].get("debug_limit", None) if debug else None
        if self.dataset_name == "flair":
            train_csv = self.config["data"]["train_csv"]
            val_csv = self.config["data"]["val_csv"]
            base_dir = self.config["data"].get("base_dir", None)

            train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data(
                rgb_to_class=FlairDataset.rgb_to_class,
                use_csv=True,
                train_csv_path=train_csv,
                val_csv_path=val_csv,
                base_dir=base_dir,
                debug_limit=self.config["data"].get("debug_limit")
            )
        else:
            train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data(
                rgb_to_class=DLRDataset.rgb_to_class,
                train_split=self.config["data"]["train_split"],
                debug_limit=debug_limit
            )

        # Apply augmentations to full images (optional, before patchifying)
        if self.config.get("augmentation"):
            train_transform = self.build_augmentation_pipeline()
            augmented_imgs, augmented_masks = [], []
            for img, mask in zip(train_imgs, train_masks):
                augmented = train_transform(image=img, mask=mask)
                augmented_imgs.append(augmented["image"])
                augmented_masks.append(augmented["mask"])
            self.logger.info(f"Augmented {len(train_imgs)} training images using Albumentations.")
            train_imgs, train_masks = augmented_imgs, augmented_masks
        else:
            train_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ], additional_targets={"mask": "mask"})

        val_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ], additional_targets={"mask": "mask"})
        self.logger.info(f"Train transform: {train_transform}")
        self.logger.info(f"Validation transform: {val_transform}")

        # Patchify both train and val sets
        def patchify_batch(imgs: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[
            List[np.ndarray], List[np.ndarray]]:
            img_patches: List[np.ndarray] = []
            mask_patches: List[np.ndarray] = []
            for img, mask in zip(imgs, masks):
                img_p, _, _ = preprocessor.patchify_image(img)
                mask_p, _, _ = preprocessor.patchify_image(mask)
                img_patches.extend(img_p)
                mask_patches.extend(mask_p)
            return img_patches, mask_patches

        train_img_patches, train_mask_patches = patchify_batch(train_imgs, train_masks)
        val_img_patches, val_mask_patches = patchify_batch(val_imgs, val_masks)

        train_dataset = self.DatasetClass(train_img_patches, train_mask_patches, transform=train_transform)
        val_dataset = self.DatasetClass(val_img_patches, val_mask_patches, transform=val_transform)

        # Create DataLoaders with collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=SatelliteDataset.collate_fn,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=SatelliteDataset.collate_fn,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory
        )

        self.logger.info(f"Patchified into {len(train_dataset)} training and {len(val_dataset)} validation patches.")
        return train_loader, val_loader, val_imgs, val_masks, preprocessor



    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Mask2FormerModel:
        """
        Initializes and trains the segmentation model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data

        Returns:
            Trained Mask2FormerModel instance
        """
        model_type = self.config["model"].get("type", "mask2former").lower()
        self.logger.info(f"Training with model type: {model_type}")
        self.logger.info("Initializing and training model...")
        if model_type == "mask2former":
            segmenter = Mask2FormerModel(
                model_name=self.pretrained_weights,
                num_classes=self.num_classes,
                accelerator=self.accelerator
            )
        elif model_type == "unet":
            segmenter = UNetResNet34Model(
                num_classes=self.num_classes,
                accelerator=self.accelerator
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        trained_model = segmenter.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            lr=self.learning_rate,
            device=self.device,
            tensorboard_writer=self.writer,
        )

        model_path: str = os.path.join(self.model_save_dir, "trained_model.pth")
        torch.save(self.accelerator.unwrap_model(trained_model).state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        return segmenter

    def evaluate(self, segmenter, val_loader: DataLoader) -> None:
        """
        Evaluates the model on validation set using standard metrics.

        Args:
            segmenter: Trained segmentation model
            val_loader: Validation DataLoader
        """
        self.logger.info("Evaluating model...")
        mean_iou, per_class_iou = segmenter.evaluate(val_loader)
        self.logger.info(f"Evaluation completed. Mean IoU: {mean_iou:.4f}, Per-Class IoU: {per_class_iou}")

    def visualize(
            self,
            segmenter: Mask2FormerModel,
            val_imgs: List[np.ndarray],
            val_masks: List[np.ndarray],
            preprocessor: DataPreprocessor,
            prefix: str = "prediction",
            max_samples: int = 3
    ) -> None:
        """
        Visualizes predictions by reconstructing full images from predicted patches.

        Args:
            segmenter: Trained segmentation model
            val_imgs: List of original validation images
            val_masks: List of original validation masks
            preprocessor: DataPreprocessor instance used for patchification/reconstruction
            prefix: Prefix for saving visualization filenames
        """
        self.logger.info(f"Generating visualizations with prefix '{prefix}'")

        for i in range(min(max_samples, len(val_imgs))):
            try:
                img = val_imgs[i]
                gt_mask = val_masks[i]
                original_shape = gt_mask.shape

                img_patches, coords, full_shape_img = preprocessor.patchify_image(img)
                mask_patches, _, full_shape_mask = preprocessor.patchify_image(gt_mask)

                self.logger.debug(f"[Visualization {i}] Original img shape: {img.shape}, padded: {full_shape_img}")
                self.logger.debug(f"[Visualization {i}] Original mask shape: {gt_mask.shape}, padded: {full_shape_mask}")

                if full_shape_img != full_shape_mask:
                    self.logger.warning(f"[Visualization {i}] Shape mismatch: {full_shape_img} vs {full_shape_mask}")
                    continue

                # Predict and reconstruct
                if self.model_type == "mask2former":
                    pred_patches = [segmenter.predict(patch, device=self.device) for patch in img_patches]
                else:
                    pred_patches = [segmenter.predict(patch) for patch in img_patches]
                pred_mask = preprocessor.reconstruct_from_patches(pred_patches, coords, full_shape_img)
                gt_mask_padded = preprocessor.reconstruct_from_patches(mask_patches, coords, full_shape_img)

                # Crop both to original shape to avoid mismatch
                H, W = original_shape
                pred_mask = pred_mask[:H, :W]
                gt_mask_padded = gt_mask_padded[:H, :W]

                # Save visualization
                save_path = os.path.join(self.visualization_dir, f"{prefix}_comparison_{i}.png")
                Visualizer.save_full_comparison(img, gt_mask_padded, pred_mask, save_path)
                self.logger.info(f"[Visualization {i}] Saved visualization: {save_path}")
                self.logger.info(f"Saving outputs to: {self.run_dir}")


            except Exception as e:
                self.logger.warning(f"[Visualization {i}] Skipping due to error: {e}")

        self.logger.info("Visualization process completed.")

    def run(self) -> None:
        try:
            train_loader, val_loader, val_imgs, val_masks, preprocessor = self.prepare_data()
            segmenter = self.train(train_loader, val_loader)
            self.evaluate(segmenter, val_loader)
            self.visualize(segmenter, val_imgs, val_masks, preprocessor, prefix="trained")
        finally:
            self.writer.close()
            self.logger.info("Workflow completed.")


if __name__ == "__main__":
    pipeline = SegmentationPipeline()
    pipeline.run()
