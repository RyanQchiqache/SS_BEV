import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from codeBase.data.DataPreprocessor import DataPreprocessor
from codeBase.models.mask2former_model import Mask2FormerModel
from codeBase.visualisation.visualizer import Visualizer

# Configuration (Directly in the script)
image_dir = "/Users/ryanqchiqache/PycharmProjects/SS_BEV/SS_data/images"
mask_dir = "/Users/ryanqchiqache/PycharmProjects/SS_BEV/SS_data/masks"
patch_size = 256
batch_size = 8
num_classes = 6
epochs = 10
learning_rate = 1e-4
pretrained_weights = "facebook/mask2former-swin-small-ade-semantic"
output_dir = "outputs"
model_save_dir = os.path.join(output_dir, "models")
visualization_dir = os.path.join(output_dir, "visualizations")
logs_dir = "logs"

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

def prepare_data():
    """Load and preprocess the dataset."""
    print("app Preparing data...")
    preprocessor = DataPreprocessor(image_dir=image_dir, mask_dir=mask_dir, patch_size=patch_size)
    train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data(train_split=0.8)
    print(f" Data prepared: {len(train_imgs)} training and {len(val_imgs)} validation samples.")
    return train_imgs, train_masks, val_imgs, val_masks

def create_dataloaders(train_imgs, train_masks, val_imgs, val_masks):
    """Create training and validation data loaders."""
    print(" Creating data loaders...")

    # Convert to PyTorch tensors and create datasets
    train_dataset = TensorDataset(
        torch.tensor(train_imgs).permute(0, 3, 1, 2).float() / 255.0,
        torch.tensor(train_masks).long()
    )
    val_dataset = TensorDataset(
        torch.tensor(val_imgs).permute(0, 3, 1, 2).float() / 255.0,
        torch.tensor(val_masks).long()
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(" Data loaders created.")
    return train_loader, val_loader

def train_model(train_loader, val_loader):
    """Train the model using the training data."""
    print(" Starting model training...")
    segmenter = Mask2FormerModel(model_name=pretrained_weights, num_classes=num_classes)

    # Train the model
    trained_model = segmenter.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device
    )

    # Save the trained model
    model_path = os.path.join(model_save_dir, "trained_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f" Model saved to {model_path}")
    return segmenter

def evaluate_model(segmenter, val_loader):
    """Evaluate the model on the validation data."""
    print("🔍 Evaluating model...")
    mean_iou, per_class_iou = segmenter.evaluate(val_loader, device=device)
    print(f" Mean IoU: {mean_iou:.4f}")
    print(f" Per-Class IoU: {per_class_iou}")

def visualize_predictions(segmenter, val_imgs, val_masks, prefix="prediction"):
    """Visualize predictions from the model."""
    print(f" Visualizing predictions: {prefix}...")
    for i in range(3):
        img = val_imgs[i]
        gt_mask = val_masks[i]
        pred_mask = segmenter.predict(img, device=device)

        # Generate comparison and save
        comparison = Visualizer.compare_masks(img, gt_mask, pred_mask)
        save_path = os.path.join(visualization_dir, f"{prefix}_comparison_{i}.png")
        comparison.save(save_path)
        print(f" Saved visualization to {save_path}")
    print(" Visualization completed.")

# Main workflow
if __name__ == "__main__":
    # Step 1: Prepare the data
    train_imgs, train_masks, val_imgs, val_masks = prepare_data()

    # Step 2: Create data loaders
    train_loader, val_loader = create_dataloaders(train_imgs, train_masks, val_imgs, val_masks)

    # Step 3: Train the model
    segmenter = train_model(train_loader, val_loader)

    # Step 4: Evaluate the trained model
    evaluate_model(segmenter, val_loader)

    # Step 5: Visualize predictions
    visualize_predictions(segmenter, val_imgs, val_masks, prefix="trained")
