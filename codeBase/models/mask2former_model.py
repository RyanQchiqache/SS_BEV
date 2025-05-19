import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
from torch.optim import AdamW


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
        # Initialize image processor with the specified model name
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name,
            reduce_labels=False,  # Do not reduce class indices (include class 0)
            do_rescale=False
        )

        # Define class label mappings if not provided
        if class_names is None:
            class_names = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled"]
        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in self.id2label.items()}

        # Load the pretrained Mask2Former model
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True  # Allow loading despite different head size
        )

        # Track whether the backbone is frozen
        self.backbone_frozen = False

    def train_model(self, train_loader, val_loader, epochs=30, lr=1e-4, device=torch.device("cpu")):
        """Train the model on the training set and evaluate on the validation set each epoch."""
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            self.model.train()
            if self.backbone_frozen:
                if hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
                    self.model.model.backbone.eval()
            total_loss = 0.0

            # Training loop
            for batch in train_loader:
                try:
                    # Unpack the batch
                    images, masks = batch

                    # Ensure correct format and move to device
                    images = images.permute(0, 2, 3, 1).contiguous().to(device)  # (B, H, W, C)
                    masks = masks.to(device)

                    # Prepare batch inputs using the Mask2FormerImageProcessor
                    batch = self.processor(images=[image for image in images],
                                           segmentation_maps=[mask for mask in masks],
                                           return_tensors="pt")
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track total loss
                    total_loss += loss.item()

                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

            avg_loss = total_loss / len(train_loader)

            # Compute validation IoU after each epoch
            val_miou = self.evaluate(val_loader, device=device)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}")

        return self.model

    def calculate_mean_iou(intersection, union):
        """Calculate mean IoU from intersection and union."""
        ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        return mean_iou, ious

    def evaluate(self, data_loader, device=torch.device("cpu")):
        """Evaluate the model on a dataset (compute mean IoU)."""
        self.model.to(device)
        self.model.eval()
        num_classes = self.model.config.num_labels if hasattr(self.model, "config") else 6

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        with torch.no_grad():
            for images, masks in data_loader:
                # Convert ground truth masks to numpy arrays
                gt_masks = np.array([np.array(m) for m in masks], dtype=np.uint8)

                # Prepare inputs and make predictions
                batch = self.prepare_batch(images, device)
                outputs = self.model(**batch)

                # Post-process to get predicted segmentation maps
                seg_maps = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[mask.shape[:2] for mask in gt_masks]
                )

                # Accumulate intersection and union for each image
                for pred_mask, true_mask in zip(seg_maps, gt_masks):
                    pred_arr = pred_mask.cpu().numpy().astype(np.uint8)
                    true_arr = true_mask.astype(np.uint8)

                    intersection, union = self.compute_iou(pred_arr, true_arr, num_classes)
                    total_intersection += intersection
                    total_union += union

        # Calculate and print mean IoU
        mean_iou, ious = self.calculate_mean_iou(total_intersection, total_union)
        print(f"Per-class IoU: {ious}")
        print(f"Mean IoU: {mean_iou:.4f}")
        return mean_iou, ious


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
