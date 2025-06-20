import os
import pytest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from codeBase.data.Preprocessing_Utild import DataPreprocessor

VISUALIZE = bool(int(os.environ.get("VISUALIZE", 0)))  # set to 1 to enable

@pytest.mark.parametrize("patch_size, overlap", [(128, 32)])
def test_patchify_and_reconstruct(patch_size, overlap):
    sample_image_path = "/home/ryqi/PycharmProjects/SS_BEV/SS_data/images/image_71.jpg"
    image = np.array(Image.open(sample_image_path).convert("RGB"))
    H_orig, W_orig = image.shape[:2]

    preprocessor = DataPreprocessor(image_dir="", mask_dir="", patch_size=patch_size, overlap=overlap)
    patches, coords, full_shape = preprocessor.patchify_image(image)
    reconstructed = preprocessor.reconstruct_from_patches(patches, coords, full_shape)

    reconstructed = reconstructed[:H_orig, :W_orig, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    diff = np.abs(image.astype(np.float32) - reconstructed.astype(np.float32))
    mean_error = np.mean(diff)
    max_error = np.max(diff)

    print(f"Mean absolute error: {mean_error:.4f}")
    print(f"Max absolute error: {max_error:.4f}")

    if VISUALIZE:
        fig, axs = plt.subplots(1, 3, figsize=(24, 12))
        axs[0].imshow(image)
        axs[0].set_title("Original", fontsize=14)
        axs[1].imshow(reconstructed)
        axs[1].set_title("Reconstructed", fontsize=14)
        axs[2].imshow(diff.astype(np.uint8))
        axs[2].set_title("Difference", fontsize=14)
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    assert mean_error < 2.0, f"Mean error too high: {mean_error}"
    assert max_error < 256.0, f"Max error too high: {max_error}"
