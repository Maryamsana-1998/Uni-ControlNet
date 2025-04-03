import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from annotator.content import ContentDetector

# Function to process a folder of images and save features
def process_folder(input_folder, output_folder):
    # Create ContentDetector instance
    detector = ContentDetector()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of image files in the input folder
    image_files = list(Path(input_folder).glob("*.png")) + list(Path(input_folder).glob("*.jpg"))

    # Process each image
    for image_file in tqdm(image_files, desc="Processing Images"):
        # Load the image
        img = cv2.imread(str(image_file))

        # Skip if image couldn't be read
        if img is None:
            print(f"Could not read image: {image_file}")
            continue

        # Get content features
        features = detector(img)

        # Save features as .npy file with the same name as the image
        output_path = Path(output_folder) / f"{image_file.stem}.npy"
        np.save(output_path, features)
        print(f"Features saved to {output_path}")


# Input and output folder paths
input_folder = "/data2/local_datasets/vimeo_all/condition/quality_8/"
output_folder = "/data2/local_datasets/vimeo_all/condition/quality_8_emb/"

# Run the processing
if __name__ == "__main__":
    process_folder(input_folder, output_folder)
