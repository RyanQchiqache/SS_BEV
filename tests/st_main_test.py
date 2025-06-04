import json
import pytest
from pathlib import Path
from codeBase.st_main import SegmentationPipeline
from codeBase.config import logging_setup  # Adjust if needed
import numpy as np
from PIL import Image

@pytest.fixture
def dummy_config(tmp_path):
    return {
        "data": {
            "images_dir": str(tmp_path / "images"),
            "masks_dir": str(tmp_path / "masks"),
            "patch_size": 128,
            "batch_size": 1,
            "num_classes": 2,
            "train_split": 0.8,
            "debug": True
        },
        "model": {
            "epochs": 1,
            "learning_rate": 0.001,
            "pretrained_weights": "nvidia/segformer-b0-finetuned-ade-512-512"
        },
        "augmentation": {
            "horizontal_flip": 0.0,
            "vertical_flip": 0.0,
            "rotate_90": 0.0
        },
        "paths": {
            "output_dir": str(tmp_path / "output"),
            "model_save_dir": str(tmp_path / "output" / "models"),
            "visualization_dir": str(tmp_path / "output" / "viz"),
            "logs_dir": str(tmp_path / "output" / "logs")
        }
    }

def test_segmentation_pipeline_end_to_end(monkeypatch, dummy_config, tmp_path):
    # Patch load_config to return dummy_config instead of reading from file
    monkeypatch.setattr(logging_setup, "load_config", lambda: dummy_config)

    # Create dummy image and mask
    images_dir = Path(dummy_config["data"]["images_dir"])
    masks_dir = Path(dummy_config["data"]["masks_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    mask = np.zeros((256, 256, 3), dtype=np.uint8)
    mask[:, :, 0] = 60  # One class color from COLOR_TO_CLASS

    Image.fromarray(img).save(images_dir / "img1.png")
    Image.fromarray(mask).save(masks_dir / "img1_mask.png")

    # Run pipeline
    pipeline = SegmentationPipeline()
    pipeline.run()

    # Check expected old_outputs
    assert (Path(dummy_config["paths"]["model_save_dir"]) / "trained_model.pth").exists()
    assert any(Path(dummy_config["paths"]["visualization_dir"]).glob("*.png")), "No visualizations generated"
