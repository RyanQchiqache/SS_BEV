from typing import Optional
import os
import torch
from torch import nn
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
from torch.optim import AdamW
from codeBase.config.logging_setup import setup_logger, load_config
from tqdm import tqdm
from accelerate import Accelerator

logger = setup_logger(__name__)

class Mask2FormerModel:
    """Wrapper for Mask2Former model to handle initialization, training, evaluation, and inference."""

    def __init__(self,
                 model_name: str = "facebook/mask2former-swin-small-ade-semantic",
                 num_classes: int = 6,
                 class_names: list = None,
                 accelerator: Optional[Accelerator] = None):
        logger.info("Loading image processor and model...")
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name,
            reduce_labels=False,
            do_rescale=False
        )
        self.config = load_config()
        model_cfg = self.config.get("model", {})

        dataset_name = model_cfg.get("dataset_name", "flair")
        label_type = model_cfg.get("label_type", None)

        if label_type is None:
            class_names = model_cfg["classes_names"].get(dataset_name)
        else:
            class_names = model_cfg["classes_names"].get(dataset_name, {}).get(label_type)

        if dataset_name == "flair" and label_type is not None:
            logger.warning("Label type was set for flair dataset, which may not use it.")

        if class_names is None:
            raise ValueError(f"Class names not found in config for dataset: {dataset_name}, label_type: {label_type}")


        logger.info(f"Loaded class names for dataset: {dataset_name}, label_type: {label_type}")
        logger.debug(f"Class names: {class_names}")

        num_classes = model_cfg.get("num_classes", len(class_names))

        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in self.id2label.items()}

        logger.debug(f"id2label mapping: {self.id2label}")

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )


        self.backbone_frozen = False
        self.accelerator = accelerator or Accelerator()
        self.model.to(self.accelerator.device)
        logger.info("Model and processor initialized successfully.")

    def train_model(self, train_loader, val_loader, epochs, lr, device=None, tensorboard_writer=None):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model, optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, optimizer, train_loader, val_loader
        )
        #best_miou = 0

        for epoch in range(1, epochs + 1):
            self._set_model_mode(train=True)
            total_loss = 0.0

            logger.info(f"Starting epoch {epoch}/{epochs}...")

            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                try:
                    loss = self._process_batch(batch, device, optimizer, batch_idx, epoch)
                    total_loss += loss
                    logger.info(f"[Epoch {epoch}] Batch {batch_idx + 1}/{len(train_loader)} processed.")
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx + 1}: {e}")
                    torch.cuda.empty_cache()

            self._log_epoch_results(epoch, total_loss, len(train_loader), val_loader, tensorboard_writer)

        return self.model

    def _set_model_mode(self, train=True):
        self.model.train() if train else self.model.eval()
        if train and self.backbone_frozen and hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            self.model.model.backbone.eval()

    def _process_batch(self, batch, device, optimizer, batch_idx, epoch):
        images, masks = batch
        #image_sample = images[0].cpu().numpy()
        #logger.info(f"[Raw Tensor] shape: {image_sample.shape}, min/max: {image_sample.min()}/{image_sample.max()}")
        images_np = [
            Mask2FormerModel.unnormalize_img(img).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            for img in images
        ]

        # Convert mask to numpy int64
        masks_np = [msk.cpu().numpy().astype(np.int64) for msk in masks]
        """if epoch == 1 and batch_idx == 0:
            img_sample = images_np[0]
            mask_sample = masks_np[0]
            logger.info(f"[Sanity Check] Image dtype: {img_sample.dtype}, shape: {img_sample.shape}, min/max: {img_sample.min()}/{img_sample.max()}")
            logger.info(f"[Sanity Check] Mask dtype: {mask_sample.dtype}, shape: {mask_sample.shape}, unique values: {np.unique(mask_sample)}")"""

        batch_inputs = self.processor(images=images_np, segmentation_maps=masks_np, return_tensors="pt")
        batch_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else [i.to(device) for i in v]) for k, v in batch_inputs.items()}

        outputs = self.model(**batch_inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        self.accelerator.backward(loss)
        optimizer.step()

        return loss.item()

    def _log_epoch_results(self, epoch, total_loss, num_batches, val_loader, writer):
        avg_loss = total_loss / num_batches
        val_miou, per_class_iou = self.evaluate(val_loader)

        logger.info(
            f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}, Per-class IoU: {per_class_iou}")

        if writer:
            writer.add_scalar("Loss/Total", avg_loss, epoch)
            writer.add_scalar("IoU/val", val_miou, epoch)
            for idx, iou in enumerate(per_class_iou):
                class_name = self.id2label.get(idx, f"Class_{idx}")
                writer.add_scalar(f"IoU/{class_name}", iou, epoch)

    @torch.no_grad()
    def evaluate(self, data_loader):
        logger.info("Evaluating model...")
        device = self.accelerator.device
        self.model.eval()
        num_classes = self.model.config.num_labels if hasattr(self.model, "config") else 6

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        for batch_idx, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            logger.info(f"Evaluating batch {batch_idx + 1}/{len(data_loader)}")
            try:
                intersection, union = self._process_eval_batch(images, masks, num_classes, device)
                total_intersection += intersection
                total_union += union
            except Exception as e:
                logger.warning(f"Error during evaluation: {e}")
                torch.cuda.empty_cache()

        mean_iou, ious = Mask2FormerModel.calculate_mean_iou(total_intersection, total_union)
        logger.info(f"Per-class IoU: {ious}")
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        return mean_iou, ious

    def _process_eval_batch(self, images, masks, num_classes, device):
        images_np = [
            Mask2FormerModel.unnormalize_img(img).permute(1, 2, 0).cpu().numpy().astype(np.float32)
            for img in images
        ]

        # Convert mask to numpy int64
        masks_np = [msk.cpu().numpy().astype(np.int64) for msk in masks]

        batch_inputs = self.processor(images=images_np, return_tensors="pt")
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = self.model(**batch_inputs)

        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[mask.shape[:2] for mask in masks_np]
        )

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        for i, (pred_mask, true_mask) in enumerate(zip(seg_maps, masks_np)):
            pred_arr = pred_mask.cpu().numpy().astype(np.uint8)
            true_arr = true_mask.astype(np.uint8)

            #print(f"[CHECK] Predicted classes in sample {i}: {np.unique(pred_arr)}")

            intersection, union = Mask2FormerModel.compute_iou(pred_arr, true_arr, num_classes)
            total_intersection += intersection
            total_union += union

        return total_intersection, total_union

    @torch.no_grad()
    def predict(self, image, device=None):
        logger.info("Generating prediction...")
        device = self.accelerator.device
        self.model.eval()

        if isinstance(image, np.ndarray):
            img_array = image.astype(np.uint8)
        else:
            img_array = image

        inputs = self.processor(images=[img_array], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        seg_map = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[img_array.shape[:2] if isinstance(img_array, np.ndarray) else (image.height, image.width)]
        )[0]

        return seg_map.cpu().numpy().astype(np.uint8)

    @staticmethod
    def calculate_mean_iou(intersection, union):
        ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        return mean_iou, ious

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
    def unnormalize_img(img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        return (img * std + mean).clamp(0, 1)

