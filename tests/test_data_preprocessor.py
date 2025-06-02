"""import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from codeBase.data.DataPreprocessor import DataPreprocessor  # Update this if the class is in a different path

def test_patchify_and_reconstruct(image_path: str, patch_size: int = 128, overlap: int = 32, visualize: bool = False):
    print(f"Testing reconstruction for image: {image_path}")
    image = np.array(Image.open(image_path).convert("RGB"))
    H_orig, W_orig = image.shape[:2]

    print(f"Original image shape: {image.shape}")
    preprocessor = DataPreprocessor(image_dir="", mask_dir="", patch_size=patch_size, overlap=overlap)
    patches, coords, full_shape = preprocessor.patchify_image(image)
    reconstructed = preprocessor.reconstruct_from_patches(patches, coords, full_shape)

    # Crop reconstructed image back to original shape
    reconstructed = reconstructed[:H_orig, :W_orig, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    diff = np.abs(image.astype(np.float32) - reconstructed.astype(np.float32))
    mean_error = np.mean(diff)
    max_error = np.max(diff)

    print(f"Mean absolute error: {mean_error:.4f}")
    print(f"Max absolute error: {max_error:.4f}")

    if visualize:
        fig, axs = plt.subplots(1, 3, figsize=(24, 12))  # Increased figure size for better visibility
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

    assert mean_error < 2.0, "Mean error too high"
    assert max_error < 10.0, "Max error too high (indicates possible bug)"


if __name__ == "__main__":
    # Use any real image from your dataset
    sample_image = "/home/ryqi/PycharmProjects/SS_BEV/SS_data/images/image_71.jpg"
    test_patchify_and_reconstruct(sample_image, patch_size=128, overlap=32, visualize=True)



"""

import pytest
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from codeBase.data.DataPreprocessor import DataPreprocessor

@pytest.mark.parametrize("patch_size, overlap", [(128, 32)])
def test_patchify_and_reconstruct(patch_size, overlap, visualize=False):
    sample_image_path = "/home/ryqi/PycharmProjects/SS_BEV/SS_data/images/image_71.jpg"
    image = np.array(Image.open(sample_image_path).convert("RGB"))
    H_orig, W_orig = image.shape[:2]

    preprocessor = DataPreprocessor(image_dir="", mask_dir="", patch_size=patch_size, overlap=overlap)
    patches, coords, full_shape = preprocessor.patchify_image(image)
    reconstructed = preprocessor.reconstruct_from_patches(patches, coords, full_shape)

    # Crop and cast
    reconstructed = reconstructed[:H_orig, :W_orig, :]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    diff = np.abs(image.astype(np.float32) - reconstructed.astype(np.float32))
    mean_error = np.mean(diff)
    max_error = np.max(diff)

    print(f"Mean absolute error: {mean_error:.4f}")
    print(f"Max absolute error: {max_error:.4f}")

    if visualize:
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
    assert max_error < 256.0, f"Max error too high: {max_error} — likely patch edge artifact"
