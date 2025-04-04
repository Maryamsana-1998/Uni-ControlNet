import os
import shutil
from PIL import Image
from tqdm import tqdm

def restructure_vimeo_sequence_in_place(base_dir):
    """
    Renames first and last images in each folder and moves intermediate images to a 'depth' subfolder (in-place).
    """
    for root, dirs, files in os.walk(base_dir):
        image_files = sorted([f for f in files if f.endswith(('.png', '.jpg'))])
        if len(image_files) < 2:
            continue

        first_img = image_files[0]
        last_img = image_files[-1]
        intermediate_imgs = image_files[1:-1]

        # Create depth folder if not exists
        depth_dir = os.path.join(root, 'depth')
        os.makedirs(depth_dir, exist_ok=True)

        # Move intermediate images to depth folder
        for img in intermediate_imgs:
            src = os.path.join(root, img)
            dst = os.path.join(depth_dir, img)
            shutil.copy(src, dst)

        # Rename first and last images
        os.rename(os.path.join(root, first_img), os.path.join(root, 'r1.png'))
        os.rename(os.path.join(root, last_img), os.path.join(root, 'r2.png'))

    return "In-place restructuring complete."

# Example usage
base_dir = '/data2/local_datasets/vimeo_septuplet/sequences'  # Adjust your path
print(restructure_vimeo_sequence_in_place(base_dir))
