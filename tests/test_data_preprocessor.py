import unittest
import os
import numpy as np
import cv2
from codeBase.data.DataPreprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create a temporary test directory and sample data."""
        cls.test_image_dir = "tests/test_images"
        cls.test_mask_dir = "tests/test_masks"
        os.makedirs(cls.test_image_dir, exist_ok=True)
        os.makedirs(cls.test_mask_dir, exist_ok=True)

        # Create a dummy image and mask (256x256 RGB)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 6, (256, 256), dtype=np.uint8)

        # Save images and masks as files
        for i in range(3):
            cv2.imwrite(f"{cls.test_image_dir}/image_{i:02d}.jpg", image)
            cv2.imwrite(f"{cls.test_mask_dir}/mask_{i:02d}.png", mask)

    @classmethod
    def tearDownClass(cls):
        """Clean up test directories after tests."""
        for folder in [cls.test_image_dir, cls.test_mask_dir]:
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
            os.rmdir(folder)

    def test_rgb_to_class(self):
        """Test conversion of RGB mask to class map."""
        preprocessor = DataPreprocessor(self.test_image_dir, self.test_mask_dir)
        rgb_mask = np.array([[[60, 16, 152], [132, 41, 246]], [[110, 193, 228], [254, 221, 58]]])
        class_map = preprocessor.rgb_to_class(rgb_mask)
        expected = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(class_map, expected)

    def test_prepare_data(self):
        """Test data preparation including patching and splitting."""
        preprocessor = DataPreprocessor(self.test_image_dir, self.test_mask_dir)
        train_imgs, train_masks, val_imgs, val_masks = preprocessor.prepare_data()

        self.assertEqual(train_imgs.shape[1:], (256, 256, 3))
        self.assertEqual(train_masks.shape[1:], (256, 256))
        self.assertGreaterEqual(len(train_imgs), 1)
        self.assertGreaterEqual(len(val_imgs), 1)


if __name__ == "__main__":
    unittest.main()
