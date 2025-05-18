import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from torch import nn
import numpy as np
from torch.optim import AdamW


class Mask2FormerModel:
    """Wrapper for Mask2Former model to handle initialization, training, evaluation, and inference."""

    def __init__(self, num_classes=6, id2label=None, label2id=None, pretrained=True):
        # Initialize image processor
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-ade-semantic",
            reduce_labels=False  # do not reduce class indices (we include class 0)
        )
        # Define class label mappings if not provided
        if id2label is None or label2id is None:
            class_names = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
            id2label = {i: name for i, name in enumerate(class_names)}
            label2id = {name: i for i, name in id2label.items()}
        # Load Mask2Former model
        if pretrained:
            if num_classes == 6:
                # Load pretrained weights and replace classification head for 6 classes:contentReference[oaicite:25]{
                # index=25}
                self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    "facebook/mask2former-swin-small-ade-semantic",
                    num_labels=num_classes,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True  # allow loading despite different head size
                )
            else:
                # Demo mode: load pre-trained model as-is (e.g., ADE20K 150 classes)
                self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    "facebook/mask2former-swin-small-ade-semantic"
                )
        else:
            # Initialize a new model from scratch (not typically used)
            self.model = Mask2FormerForUniversalSegmentation.from_config(
                self.model.config if hasattr(self, 'model') else {}
            )
        self.backbone_frozen = False  # track backbone freeze state


    def train_model(self, train_loader, val_loader, epochs=5, lr=1e-4, device=torch.device("cpu")):
        """Train the model on the training set and evaluate on the validation set each epoch."""
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            self.model.train()
            # If backbone is frozen, ensure it stays in eval mode during training
            if self.backbone_frozen:
                if hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
                    self.model.model.backbone.eval()
            total_loss = 0.0
            # Training loop
            for images, masks in train_loader:
                # Prepare batch inputs using the Mask2FormerImageProcessor
                batch = self.processor(images=images, segmentation_maps=masks, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            # Compute validation IoU after each epoch
            val_miou = self.evaluate(val_loader, device=device)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}")
        return self.model

    def evaluate(self, data_loader, device=torch.device("cpu")):
        """Evaluate the model on a dataset (compute mean IoU)."""
        self.model.to(device)
        self.model.eval()
        num_classes = self.model.config.num_labels if hasattr(self.model, "config") else 6
        # Initialize confusion matrix components
        intersection = [0] * num_classes
        union = [0] * num_classes
        with torch.no_grad():
            for images, masks in data_loader:
                # Move ground truth masks to numpy (if not already) for evaluation
                gt_masks = [np.array(m) for m in masks]
                # Prepare inputs (images only, for prediction)
                batch = self.processor(images=images, return_tensors="pt")
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                # Get predicted segmentation maps (list of tensors, one per image)
                seg_maps = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[mask.shape[:2] for mask in gt_masks]
                )
                # Calculate intersection and union for each class
                for pred_mask, true_mask in zip(seg_maps, gt_masks):
                    pred_arr = pred_mask.cpu().numpy().astype(np.uint8)
                    true_arr = true_mask.astype(np.uint8)
                    for c in range(num_classes):
                        # Skip classes not present in either to avoid dividing by zero
                        pred_c = (pred_arr == c)
                        true_c = (true_arr == c)
                        if np.any(pred_c) or np.any(true_c):
                            inter = np.logical_and(pred_c, true_c).sum()
                            union_c = np.logical_or(pred_c, true_c).sum()
                            intersection[c] += inter
                            union[c] += union_c
        # Compute mean IoU over classes (avoid division by zero)
        ious = []
        for c in range(num_classes):
            if union[c] > 0:
                ious.append(intersection[c] / union[c])
        mean_iou = np.mean(ious) if ious else 0.0
        return mean_iou

    def predict(self, image, device=torch.device("cpu")):
        """Run inference on a single image (or patch). Returns the predicted segmentation map (H×W)."""
        self.model.to(device)
        self.model.eval()
        # If input is a NumPy array, convert to PIL Image for processor compatibility
        if isinstance(image, np.ndarray):
            # Ensure RGB order and uint8 dtype
            img_array = image.astype(np.uint8)
        else:
            img_array = image  # assume PIL or already suitable
        inputs = self.processor(images=[img_array], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        seg_map = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[img_array.shape[:2] if isinstance(img_array, np.ndarray) else (image.height, image.width)]
        )[0]
        return seg_map.cpu().numpy().astype(np.uint8)
