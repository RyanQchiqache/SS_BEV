import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from codeBase.config.logging_setup import setup_logger
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Tuple, List, Union

logger = setup_logger(__name__)


class Mask2FormerModel:
    """Wrapper for Mask2Former model to handle initialization, training, evaluation, and inference."""

    def __init__(self,
                 model_name: str = "facebook/mask2former-swin-small-ade-semantic",
                 num_classes: int = 6,
                 class_names: list = None):
        logger.info("Loading image processor and model...")
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name,
            reduce_labels=False,
            do_rescale=False
        )

        if class_names is None:
            class_names = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in self.id2label.items()}

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        self.backbone_frozen = False
        logger.info("Model and processor initialized successfully.")

    def train_model(self,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    epochs: int,
                    lr: float,
                    device: torch.device = torch.device("cpu"),
                    tensorboard_writer: Optional[SummaryWriter] = None,
                    use_amp: bool=False) -> torch.nn.Module:
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler = GradScaler()

        for epoch in range(1, epochs + 1):
            self._set_model_mode(train=True)
            total_loss = 0.0

            logger.info(f"Starting epoch {epoch}/{epochs}...")

            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                try:
                    loss = self._process_batch(batch, device, use_amp, optimizer, scaler)
                    total_loss += loss
                    logger.info(f"[Epoch {epoch}] Batch {batch_idx + 1}/{len(train_loader)} processed.")
                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
                    torch.cuda.empty_cache()

            self._log_epoch_results(epoch, total_loss, len(train_loader), val_loader, tensorboard_writer)

        return self.model

    def _set_model_mode(self, train: bool=True)-> None:
        self.model.train() if train else self.model.eval()
        if train and self.backbone_frozen and hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            self.model.model.backbone.eval()

    def _process_batch(self,
                       batch: Tuple[torch.Tensor, torch.Tensor],
                       device: torch.device,
                       use_amp: bool,
                       optimizer: torch.optim.Optimizer,
                       scaler: torch.cuda.amp.GradScaler) -> float:
        images, masks = batch
        images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
        masks_np = [msk.cpu().numpy() for msk in masks]

        batch_inputs = self.processor(images=images_np, segmentation_maps=masks_np, return_tensors="pt")
        batch_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else [i.to(device) for i in v]) for k, v in
                        batch_inputs.items()}

        with autocast(enabled=use_amp):
            outputs = self.model(**batch_inputs)
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    def _log_epoch_results(self,
                           epoch: int,
                           total_loss: float,
                           num_batches: int,
                           val_loader: DataLoader,
                           writer: Optional[SummaryWriter]) -> None:
        avg_loss = total_loss / num_batches
        val_miou, per_class_iou = self.evaluate(val_loader)

        logger.info(
            f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}, Per-class IoU: {per_class_iou}")

        if writer:
            writer.add_scalar("Loss/Total", avg_loss, epoch)
            writer.add_scalar("IoU/val", val_miou, epoch)
            for idx, iou in enumerate(per_class_iou):
                writer.add_scalar(f"IoU/Class_{idx}", iou, epoch)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, device: torch.device=torch.device("cpu"))-> Tuple[float, np.ndarray]:
        logger.info("Evaluating model...")
        self.model.to(device)
        self.model.eval()
        num_classes = self.model.config.num_labels if hasattr(self.model, "config") else 6

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        with torch.no_grad():
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

    def _process_eval_batch(self,
                            images: List[torch.Tensor],
                            masks: List[torch.Tensor],
                            num_classes: int,
                            device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        images_np = [img.permute(1, 2, 0).contiguous().cpu().numpy() for img in images]
        masks_np = [msk.cpu().numpy() for msk in masks]

        batch_inputs = self.processor(images=images_np, return_tensors="pt")
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        outputs = self.model(**batch_inputs)

        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[mask.shape[:2] for mask in masks_np]
        )

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        for pred_mask, true_mask in zip(seg_maps, masks_np):
            pred_arr = pred_mask.cpu().numpy().astype(np.uint8)
            true_arr = true_mask.astype(np.uint8)

            intersection, union = Mask2FormerModel.compute_iou(pred_arr, true_arr, num_classes)
            total_intersection += intersection
            total_union += union

        return total_intersection, total_union

    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, torch.Tensor], device=torch.device("cpu")) -> np.ndarray:
        logger.info("Generating prediction...")
        self.model.to(device)
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
    def calculate_mean_iou(intersection: np.ndarray, union: np.ndarray)-> Tuple[float, np.ndarray]:
        ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        return mean_iou, ious

    @staticmethod
    def compute_iou(pred: np.ndarray, true: np.ndarray, num_classes:int)-> Tuple[np.ndarray, np.ndarray]:
        intersection = np.zeros(num_classes, dtype=np.int64)
        union = np.zeros(num_classes, dtype=np.int64)

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (true == cls)

            intersection[cls] = np.logical_and(pred_mask, true_mask).sum()
            union[cls] = np.logical_or(pred_mask, true_mask).sum()

        return intersection, union
