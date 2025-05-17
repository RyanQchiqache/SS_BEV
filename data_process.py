import os
import shutil

root_directory = 'Semantic_segmentation_dataset/'
image_target_dir = 'SS_BEV/SS_data/images/'
mask_target_dir = 'SS_BEV/SS_data/masks/'

os.makedirs(image_target_dir, exist_ok=True)
os.makedirs(mask_target_dir, exist_ok=True)

image_count = 1

# Loop through each tile folder (assuming 1 to 8)
for tile_num in range(1, 9):
    tile_folder = os.path.join(root_directory, f'Tile {tile_num}')
    image_folder = os.path.join(tile_folder, 'images')
    mask_folder = os.path.join(tile_folder, 'masks')

    print(f"Looking in tile folder: {tile_folder}")

    if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
        print(f" Folder not found: {image_folder} or {mask_folder}")
        continue

    images = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    masks = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    if len(images) != len(masks):
        print(f" Mismatch between images and masks in {tile_folder}")
        continue

    for image_file, mask_file in zip(images, masks):

        formatted_count = f"{image_count:02d}"

        # Move and rename image
        src_image_path = os.path.join(image_folder, image_file)
        dst_image_path = os.path.join(image_target_dir, f'image_{formatted_count}.jpg')

        try:
            shutil.move(src_image_path, dst_image_path)
            print(f' Moved and renamed image: {src_image_path} -> {dst_image_path}')
        except Exception as e:
            print(f" Failed to move image: {src_image_path} -> {dst_image_path}, Error: {e}")

        # Move and rename corresponding mask
        src_mask_path = os.path.join(mask_folder, mask_file)
        dst_mask_path = os.path.join(mask_target_dir, f'mask_{formatted_count}.png')

        try:
            shutil.move(src_mask_path, dst_mask_path)
            print(f' Moved and renamed mask: {src_mask_path} -> {dst_mask_path}')
        except Exception as e:
            print(f" Failed to move mask: {src_mask_path} -> {dst_mask_path}, Error: {e}")

        image_count += 1

print(" All images and masks have been consolidated and renamed successfully.")
