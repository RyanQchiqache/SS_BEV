import os
import torch
import numpy as np
import albumentations as A
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from codeBase.config.logging_setup import setup_logger, load_config
from codeBase.data.DataPreprocessor import DataPreprocessor
from codeBase.models.mask2former_model import Mask2FormerModel
from codeBase.visualisation.visualizer import Visualizer
from codeBase.data.satelite_dataset import SatelliteDataset

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

        self.image_dir: str = self.config["data"]["images_dir"]
        self.mask_dir: str = self.config["data"]["masks_dir"]
        self.patch_size: int = int(self.config["data"]["patch_size"])
        self.batch_size: int = int(self.config["data"]["batch_size"])
        self.num_classes: int = int(self.config["data"]["num_classes"])
        self.epochs: int = int(self.config["model"]["epochs"])
        self.learning_rate: float = float(self.config["model"]["learning_rate"])
        self.pretrained_weights: str = self.config["model"]["pretrained_weights"]

        self.output_dir: str = self.config["paths"]["output_dir"]
        self.model_save_dir: str = self.config["paths"]["model_save_dir"]
        self.visualization_dir: str = self.config["paths"]["visualization_dir"]
        self.logs_dir: str = self.config["paths"]["logs_dir"]
        self.tensorboard_dir: str = os.path.join(self.logs_dir, "tensorboard")

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        for dir_path in [self.output_dir, self.model_save_dir, self.visualization_dir, self.logs_dir, self.tensorboard_dir]:
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

        return A.Compose(transforms_list, additional_targets={"mask": "mask"})

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, List[np.ndarray], List[np.ndarray], DataPreprocessor]:
        """
        Loads and patchifies data, applies augmentations, and prepares data loaders.

        Returns:
            Tuple containing training and validation DataLoaders, original val images/masks, and the preprocessor
        """
        self.logger.info("Preparing data...")
        preprocessor = DataPreprocessor(image_dir=self.image_dir, mask_dir=self.mask_dir, patch_size=self.patch_size)
        train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data(
            train_split=self.config["data"]["train_split"],
            debug_limit=100
        )

        def patchify_batch(imgs: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

        transform = self.build_augmentation_pipeline()

        train_dataset = SatelliteDataset(train_img_patches, train_mask_patches, transform=transform)
        val_dataset = SatelliteDataset(val_img_patches, val_mask_patches, transform=None)

        train_loader: DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=SatelliteDataset.collate_fn)
        val_loader: DataLoader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=SatelliteDataset.collate_fn)

        self.logger.info(f"Patchified into {len(train_img_patches)} training and {len(val_img_patches)} validation patches.")
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
        self.logger.info("Initializing and training model...")
        segmenter = Mask2FormerModel(model_name=self.pretrained_weights, num_classes=self.num_classes)

        trained_model = segmenter.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            lr=self.learning_rate,
            device=self.device,
            tensorboard_writer=self.writer,
        )

        model_path: str = os.path.join(self.model_save_dir, "trained_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        return segmenter

    def evaluate(self, segmenter: Mask2FormerModel, val_loader: DataLoader) -> None:
        """
        Evaluates the model on validation set using standard metrics.

        Args:
            segmenter: Trained segmentation model
            val_loader: Validation DataLoader
        """
        self.logger.info("Evaluating model...")
        mean_iou, per_class_iou = segmenter.evaluate(val_loader, device=self.device)
        self.logger.info(f"Evaluation completed. Mean IoU: {mean_iou:.4f}, Per-Class IoU: {per_class_iou}")

    def visualize(self, segmenter: Mask2FormerModel, val_imgs: List[np.ndarray], val_masks: List[np.ndarray], preprocessor: DataPreprocessor, prefix: str = "prediction") -> None:
        """
        Visualizes predictions by reconstructing full images from predicted patches.

        Args:
            segmenter: Trained segmentation model
            val_imgs: List of original validation images
            val_masks: List of original validation masks
            preprocessor: DataPreprocessor instance used to reconstruct
            prefix: Prefix for saving visualizations
        """
        self.logger.info(f"Generating visualizations with prefix '{prefix}'")
        for i in range(min(3, len(val_imgs))):
            try:
                img = val_imgs[i]
                gt_mask = val_masks[i]
                img_patches, coords, full_shape = preprocessor.patchify_image(img)
                pred_patches = [segmenter.predict(patch, device=self.device) for patch in img_patches]
                pred_mask = preprocessor.reconstruct_from_patches(pred_patches, coords, full_shape)
                save_path: str = os.path.join(self.visualization_dir, f"{prefix}_comparison_{i}.png")
                Visualizer.save_full_comparison(img, gt_mask, pred_mask, save_path)
                self.logger.info(f"Saved visualization: {save_path}")
            except Exception as e:
                self.logger.warning(f"Skipping visualization {i} due to error: {e}")
        self.logger.info("Visualization process completed.")

    def run(self) -> None:
        """Executes the full training and evaluation workflow."""
        train_loader, val_loader, val_imgs, val_masks, preprocessor = self.prepare_data()
        segmenter = self.train(train_loader, val_loader)
        self.evaluate(segmenter, val_loader)
        self.visualize(segmenter, val_imgs, val_masks, preprocessor, prefix="trained")
        self.writer.close()
        self.logger.info("Workflow completed.")

if __name__ == "__main__":
    pipeline = SegmentationPipeline()
    pipeline.run()
