import os
import numpy as np
import torch
import cv2
import sys
from tqdm import tqdm
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def generate_mask_image(image_path, sam_checkpoint, model_type, device="cuda"):
    # Load and resize image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))  # Resize to 512x512
    
    # Load model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    
    # Create a plain background image
    plain_background = np.ones_like(image) * 255
    
    # Apply masks to the plain background
    for ann in masks:
        mask = ann['segmentation']
        color = np.random.randint(0, 255, size=(1, 3), dtype=np.uint8)
        plain_background[mask] = color
    
    return plain_background

def process_images(input_dir, output_dir, sam_checkpoint, model_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of already processed images
    processed_images = {img_filename for img_filename in os.listdir(output_dir)}

    # Get list of images to process
    images_to_process = [img_filename for img_filename in os.listdir(input_dir) if (img_filename.endswith('.png') or img_filename.endswith('.jpg')) and img_filename not in processed_images]
    
    print(f"Total images to process: {len(images_to_process)}")

    # Process each image in the input directory
    for img_filename in tqdm(images_to_process, desc="Processing images"):
        image_path = os.path.join(input_dir, img_filename)
        generated_image = generate_mask_image(image_path, sam_checkpoint, model_type)
        
        # Save the generated mask image
        output_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(output_path, cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR))
        print(f"Saved mask image for {img_filename} to {output_path}")

# Example usage
input_dir = '/data/maryam.sana/Uni-ControlNet/data/vimeo_data/vimeo_images'
output_dir = '/data/maryam.sana/Uni-ControlNet/data/vimeo_data/vimeo_conditions/segSAM'
sam_checkpoint = "/data/maryam.sana/NVC_image_captioning/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

process_images(input_dir, output_dir, sam_checkpoint, model_type)
