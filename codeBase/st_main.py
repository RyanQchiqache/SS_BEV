import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import logging

from codeBase.data.DataPreprocessor import DataPreprocessor
from codeBase.models.mask2former_model import Mask2FormerModel
from codeBase.visualisation.visualizer import Visualizer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
image_dir = "/home/ryqi/PycharmProjects/SS_BEV/SS_BEV/images"
mask_dir = "/home/ryqi/PycharmProjects/SS_BEV/SS_BEV/masks"
patch_size = 8
batch_size = 1
num_classes = 6
epochs = 1
learning_rate = 1e-4
pretrained_weights = "facebook/mask2former-swin-small-ade-semantic"
output_dir = "outputs"
model_save_dir = os.path.join(output_dir, "models")
visualization_dir = os.path.join(output_dir, "visualizations")
logs_dir = "logs"
tensorboard_dir = os.path.join(logs_dir, "tensorboard")

# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

# Logging configuration
logging.basicConfig(
    filename=os.path.join(logs_dir, "training.log"),
    filemode='w',
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

# TensorBoard setup
writer = SummaryWriter(log_dir=tensorboard_dir)

def prepare_data():
    logger.info("Preparing data...")
    preprocessor = DataPreprocessor(image_dir=image_dir, mask_dir=mask_dir, patch_size=patch_size)
    train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data(train_split=0.8)
    logger.info(f"Loaded {len(train_imgs)} training and {len(val_imgs)} validation samples.")
    return train_imgs, train_masks, val_imgs, val_masks

def create_dataloaders(train_imgs, train_masks, val_imgs, val_masks):
    print("[INFO] Creating data loaders...")
    logger.info("Creating data loaders...")

    train_dataset = TensorDataset(
        torch.tensor(train_imgs).permute(0, 3, 1, 2).float() / 255.0,
        torch.tensor(train_masks).long()
    )
    val_dataset = TensorDataset(
        torch.tensor(val_imgs).permute(0, 3, 1, 2).float() / 255.0,
        torch.tensor(val_masks).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Data loaders created.")
    return train_loader, val_loader

def train_model(train_loader, val_loader):
    logger.info("Initializing and training model...")
    segmenter = Mask2FormerModel(model_name=pretrained_weights, num_classes=num_classes)

    trained_model = segmenter.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        tensorboard_writer=writer,
    )

    model_path = os.path.join(model_save_dir, "trained_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    return segmenter

def evaluate_model(segmenter, val_loader):
    print("[INFO] Evaluating model on validation data...")
    logger.info("Evaluating model...")
    mean_iou, per_class_iou = segmenter.evaluate(val_loader, device=device)
    logger.info(f"Evaluation completed. Mean IoU: {mean_iou:.4f}, Per-Class IoU: {per_class_iou}")

def visualize_predictions(segmenter, val_imgs, val_masks, prefix="prediction"):
    logger.info(f"Generating visualizations with prefix '{prefix}'")
    for i in range(min(3, len(val_imgs))):
        try:
            logger.info(f"Processing visualization {i + 1} of {min(3, len(val_imgs))}...")
            img = val_imgs[i]
            gt_mask = val_masks[i]
            pred_mask = segmenter.predict(img, device=device)

            comparison = Visualizer.compare_masks(img, gt_mask, pred_mask)
            save_path = os.path.join(visualization_dir, f"{prefix}_comparison_{i}.png")
            comparison.save(save_path)
            logger.info(f"Saved visualization: {save_path}")
        except Exception as e:
            logger.warning(f"Skipping visualization {i} due to error: {e}")
    logger.info("Visualization process completed.")

# Main workflow
if __name__ == "__main__":
    train_imgs, train_masks, val_imgs, val_masks = prepare_data()
    train_loader, val_loader = create_dataloaders(train_imgs, train_masks, val_imgs, val_masks)
    segmenter = train_model(train_loader, val_loader)
    evaluate_model(segmenter, val_loader)
    visualize_predictions(segmenter, val_imgs, val_masks, prefix="trained")
    writer.close()
    logger.info("Workflow completed.")
