import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
from torch.optim import AdamW
from codeBase.config.logging_setup import setup_logger

logger = setup_logger(__name__)

class Mask2FormerModel:
    """Wrapper for Mask2Former model to handle initialization, training, evaluation, and inference."""

    def __init__(self,
                 model_name: str = "facebook/mask2former-swin-small-ade-semantic",
                 num_classes: int = 6,
                 class_names: list = None):
        """
        Initialize the Mask2Former model for universal segmentation.

        Parameters:
            model_name (str): Name of the pretrained model from Hugging Face.
            num_classes (int): Number of segmentation classes.
            class_names (list): Optional list of class names. If not provided, uses default.
        """
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

    def train_model(self, train_loader, val_loader, epochs, lr, device=torch.device("cpu"), tensorboard_writer=None):
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            self.model.train()
            if self.backbone_frozen and hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
                self.model.model.backbone.eval()
            total_loss = 0.0

            logger.info(f"Starting epoch {epoch}/{epochs}...")

            for batch in train_loader:
                try:
                    images, masks = batch
                    images_np = [img.permute(1, 2, 0).contiguous().cpu().numpy() for img in images]
                    masks_np = [msk.cpu().numpy() for msk in masks]

                    batch_inputs = self.processor(
                        images=images_np,
                        segmentation_maps=masks_np,
                        return_tensors="pt"
                    )
                    for k, v in batch_inputs.items():
                        if isinstance(v, torch.Tensor):
                            batch_inputs[k] = v.to(device)
                        elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                            batch_inputs[k] = [i.to(device) for i in v]

                    outputs = self.model(**batch_inputs)
                    loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                except Exception as e:
                    logger.warning(f"Error processing batch: {e}")
                    torch.cuda.empty_cache()
                    continue

            avg_loss = total_loss / len(train_loader)
            val_miou, per_class_iou = self.evaluate(val_loader, device=device)
            logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}, Per-class IoU: {per_class_iou}")

            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train", avg_loss, epoch)
                tensorboard_writer.add_scalar("IoU/val", val_miou, epoch)

        return self.model
    @staticmethod
    def calculate_mean_iou(intersection, union):
        ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        return mean_iou, ious

    def evaluate(self, data_loader, device=torch.device("cpu")):
        logger.info("Evaluating model...")
        self.model.to(device)
        self.model.eval()
        num_classes = self.model.config.num_labels if hasattr(self.model, "config") else 6

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        with torch.no_grad():
            for images, masks in data_loader:
                try:
                    images_np = [img.permute(1, 2, 0).contiguous().cpu().numpy() for img in images]
                    masks_np = [msk.cpu().numpy() for msk in masks]

                    batch_inputs = self.processor(images=images_np, return_tensors="pt")
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    outputs = self.model(**batch_inputs)

                    seg_maps = self.processor.post_process_semantic_segmentation(
                        outputs, target_sizes=[mask.shape[:2] for mask in masks_np]
                    )

                    for pred_mask, true_mask in zip(seg_maps, masks_np):
                        pred_arr = pred_mask.cpu().numpy().astype(np.uint8)
                        true_arr = true_mask.astype(np.uint8)

                        intersection, union = Mask2FormerModel.compute_iou(pred_arr, true_arr, num_classes)
                        total_intersection += intersection
                        total_union += union

                except Exception as e:
                    logger.warning(f"Error during evaluation: {e}")
                    torch.cuda.empty_cache()
                    continue

        mean_iou, ious = Mask2FormerModel.calculate_mean_iou(total_intersection, total_union)
        logger.info(f"Per-class IoU: {ious}")
        logger.info(f"Mean IoU: {mean_iou:.4f}")
        return mean_iou, ious

    def predict(self, image, device=torch.device("cpu")):
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
    def compute_iou(pred, true, num_classes):
        intersection = np.zeros(num_classes, dtype=np.int64)
        union = np.zeros(num_classes, dtype=np.int64)

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (true == cls)

            intersection[cls] = np.logical_and(pred_mask, true_mask).sum()
            union[cls] = np.logical_or(pred_mask, true_mask).sum()

        return intersection, union
