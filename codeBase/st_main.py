import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from codeBase.config.logging_setup import setup_logger
from codeBase.config.logging_setup import load_config


from codeBase.data.DataPreprocessor import DataPreprocessor
from codeBase.models.mask2former_model import Mask2FormerModel
from codeBase.visualisation.visualizer import Visualizer

config = load_config()

torch.manual_seed(42)
np.random.seed(42)

image_dir = config["data"]["images_dir"]
mask_dir = config["data"]["masks_dir"]
patch_size = int(config["data"]["patch_size"])
batch_size = int(config["data"]["batch_size"])
num_classes = int(config["data"]["num_classes"])
epochs = int(config["model"]["epochs"])
learning_rate = float(config["model"]["learning_rate"])
pretrained_weights = config["model"]["pretrained_weights"]

output_dir = config["paths"]["output_dir"]
model_save_dir = config["paths"]["model_save_dir"]
visualization_dir = config["paths"]["visualization_dir"]
logs_dir = config["paths"]["logs_dir"]
tensorboard_dir = os.path.join(logs_dir, "tensorboard")


# CUDA setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for dir_path in [output_dir, model_save_dir, visualization_dir, logs_dir, tensorboard_dir]:
    os.makedirs(dir_path, exist_ok=True)

logger = setup_logger(__name__)
writer = SummaryWriter(log_dir=tensorboard_dir)

def prepare_data():
    logger.info("Preparing data...")
    preprocessor = DataPreprocessor(image_dir=image_dir, mask_dir=mask_dir, patch_size=patch_size)
    train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data(
        train_split=config["data"]["train_split"],
        debug_limit=100
    )
    logger.info(f"Loaded {len(train_imgs)} training and {len(val_imgs)} validation samples.")
    return train_imgs, train_masks, val_imgs, val_masks

def create_dataloaders(train_imgs, train_masks, val_imgs, val_masks):
    logger.info("Creating data loaders...")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_tensors = [(transform(img), torch.tensor(mask).long()) for img, mask in zip(train_imgs, train_masks)]
    val_tensors = [(transform(img), torch.tensor(mask).long()) for img, mask in zip(val_imgs, val_masks)]

    train_loader = DataLoader(train_tensors, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensors, batch_size=batch_size, shuffle=False)

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
            logger.info(f"Image shape: {img.shape}, Ground truth shape: {gt_mask.shape}")

            pred_mask = segmenter.predict(img, device=device)

            save_path = os.path.join(visualization_dir, f"{prefix}_comparison_{i}.png")
            Visualizer.save_full_comparison(img, gt_mask, pred_mask, save_path)
            logger.info(f"Saved visualization: {save_path}")
        except Exception as e:
            logger.warning(f"Skipping visualization {i} due to error: {e}")
    logger.info("Visualization process completed.")


if __name__ == "__main__":
    train_imgs, train_masks, val_imgs, val_masks = prepare_data()
    train_loader, val_loader = create_dataloaders(train_imgs, train_masks, val_imgs, val_masks)
    segmenter = train_model(train_loader, val_loader)
    evaluate_model(segmenter, val_loader)
    visualize_predictions(
        segmenter,
        val_imgs,
        val_masks,
        prefix="trained"
    )
    writer.close()
    logger.info("Workflow completed.")
