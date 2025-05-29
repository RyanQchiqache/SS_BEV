from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from codeBase.config.logging_setup import setup_logger

logger = setup_logger(__name__)

class Visualizer:
    """
    Utility class for visualizing segmentation results.
    Provides overlaying and comparison of ground truth and predicted masks.
    """
    # Updated class index to visualization color mapping (in RGB for PIL and OpenCV)
    CLASS_TO_COLOR = {
        0: (60, 16, 152),  # Building (purple)
        1: (132, 41, 246),  # Land (violet)
        2: (110, 193, 228),  # Road (sky blue)
        3: (254, 221, 58),  # Vegetation (yellow)
        4: (226, 169, 41),  # Water (mustard)
        5: (155, 155, 155)  # Unlabeled (gray)
    }

    @staticmethod
    def overlay_prediction(image, mask, alpha=0.6):
        """
        Overlay a segmentation mask on an image.
        image: input image (NumPy array or PIL Image).
        mask: 2D array of class indices.
        alpha: transparency for the overlay (0=only image, 1=only mask).
        Returns a PIL Image with overlay.
        """
        # Ensure the image is in RGB format
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        else:
            image = image.copy()

        overlay = image.copy()
        H, W = mask.shape

        # Apply the visualization colors to the overlay
        for class_idx, color in Visualizer.CLASS_TO_COLOR.items():
            overlay[mask == class_idx] = color

        # Blend the original image and the colored overlay
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        return Image.fromarray(blended)

    @staticmethod
    def compare_masks(image, true_mask, pred_mask, alpha=0.5):
        """
        Create a side-by-side comparison of the ground truth mask and predicted mask over the image.
        image: input image (NumPy array or PIL Image).
        true_mask: ground truth mask (2D array of class indices).
        pred_mask: predicted mask (2D array of class indices).
        alpha: transparency for the overlay.
        Returns a PIL image showing both overlays side by side.
        """
        # Generate overlay images for both ground truth and prediction
        vis_true = Visualizer.overlay_prediction(image, true_mask, alpha=alpha)
        vis_pred = Visualizer.overlay_prediction(image, pred_mask, alpha=alpha)

        # Combine both images horizontally
        combined_width = vis_true.width + vis_pred.width
        combined_height = max(vis_true.height, vis_pred.height)
        comparison = Image.new("RGB", (combined_width, combined_height))

        # Paste both images side by side
        comparison.paste(vis_true, (0, 0))
        comparison.paste(vis_pred, (vis_true.width, 0))

        return comparison

    @staticmethod
    def save_overlay(image, mask, filepath, alpha=0.6):
        """
        Save the overlay of an image and mask to a specified filepath.
        """
        overlay = Visualizer.overlay_prediction(image, mask, alpha=alpha)
        overlay.save(filepath)
        logger.info(f"Saved overlay to {filepath}")


    @staticmethod
    def save_comparison(image, true_mask, pred_mask, filepath, alpha=0.5):
        """
        Save a side-by-side comparison of ground truth and predicted masks.
        """
        comparison = Visualizer.compare_masks(image, true_mask, pred_mask, alpha=alpha)
        comparison.save(filepath)
        logger.info(f" Saved comparison to {filepath}")

    @staticmethod
    def save_full_comparison(image, gt_mask, pred_mask, save_path, alpha=0.6):
        """
        Save a subplot comparison: [Original Image] [GT Overlay] [Pred Overlay]
        """
        # Prepare overlays
        gt_overlay = Visualizer.overlay_prediction(image, gt_mask, alpha)
        pred_overlay = Visualizer.overlay_prediction(image, pred_mask, alpha)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert PIL images to displayable format
        orig_image = np.array(image if isinstance(image, Image.Image) else Image.fromarray(image))

        axes[0].imshow(orig_image)
        axes[0].set_title("Original Image")

        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth Overlay")

        axes[2].imshow(pred_overlay)
        axes[2].set_title("Predicted Overlay")

        # Clean up axes
        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved full comparison to {save_path}")

