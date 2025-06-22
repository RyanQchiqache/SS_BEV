import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch.optim import Adam
from tqdm import tqdm
from accelerate import Accelerator
import segmentation_models_pytorch as smp
from codeBase.config.logging_setup import setup_logger, load_config

logger = setup_logger(__name__)

class UNetResNet34Model:
    def __init__(self, num_classes: int = 6, class_names: list = None, accelerator: Optional[Accelerator] = None):
        logger.info("Initializing U-Net with ResNet34 encoder...")
        self.config = load_config()
        model_cfg = self.config.get("model", {})
        dataset_name = model_cfg.get("dataset_name", "flair")
        label_type = model_cfg.get("label_type", None)

        if label_type is None:
            class_names = model_cfg["classes_names"].get(dataset_name)
        else:
            class_names = model_cfg["classes_names"].get(dataset_name, {}).get(label_type)

        if class_names is None:
            raise ValueError(f"Class names not found in config for dataset: {dataset_name}, label_type: {label_type}")

        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in self.id2label.items()}
        num_classes = model_cfg.get("num_classes", len(class_names))
        self.num_classes = num_classes

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )

        self.accelerator = accelerator or Accelerator()
        self.model.to(self.accelerator.device)
        logger.info("U-Net model initialized successfully.")

    def train_model(self, train_loader, val_loader, epochs, lr, tensorboard_writer=None):
        optimizer = Adam(self.model.parameters(), lr=lr)
        device = self.accelerator.device
        criterion = nn.CrossEntropyLoss()

        self.model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, optimizer, train_loader, val_loader
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            logger.info(f"Starting epoch {epoch}/{epochs}...")

            for batch_idx, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader)):
                images, masks = images.to(self.accelerator.device), masks.to(self.accelerator.device)

                outputs = self.model(images)
                loss = criterion(outputs, masks.long())

                optimizer.zero_grad()
                self.accelerator.backward(loss)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_miou, per_class_iou = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}")
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/Train", avg_loss, epoch)
                tensorboard_writer.add_scalar("IoU/Val", val_miou, epoch)
                for idx, iou in enumerate(per_class_iou):
                    tensorboard_writer.add_scalar(f"IoU/Class_{idx}", iou, epoch)

        return self.model

    @torch.no_grad()
    def evaluate(self, data_loader):
        logger.info("Evaluating U-Net model...")
        device = self.accelerator.device
        self.model.eval()
        num_classes = self.num_classes

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for pred, true in zip(preds_np, masks_np):
                intersection, union = self.compute_iou(pred, true, num_classes)
                total_intersection += intersection
                total_union += union

        mean_iou, per_class_iou = self.calculate_mean_iou(total_intersection, total_union)
        return mean_iou, per_class_iou

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """Takes a single preprocessed image tensor (C, H, W) and returns predicted mask as numpy array."""
        self.model.eval()
        device = self.accelerator.device
        image = image.unsqueeze(0).to(device)
        output = self.model(image)
        pred = torch.argmax(output, dim=1)[0]
        return pred.cpu().numpy()

    @staticmethod
    def compute_iou(pred, true, num_classes):
        intersection = np.zeros(num_classes, dtype=np.int64)
        union = np.zeros(num_classes, dtype=np.int64)

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (true == cls)
            intersection[cls] = np.logical_and(pred_mask, true_mask).sum()
            union[cls] = np.logical_or(pred_mask, true_mask).sum()

        return intersection, union

    @staticmethod
    def calculate_mean_iou(intersection, union):
        ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        return mean_iou, ious
