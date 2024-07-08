import os
import numpy as np
from PIL import Image
from annotator.content import ContentDetector

# Initialize the ContentDetector
apply_content = ContentDetector()

def process_image(image_path):
    """
    Process an image and extract its content embedding.
    
    Args:
    - image_path (str): Path to the image file.
    
    Returns:
    - np.ndarray: The content embedding of the image.
    """
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    content_emb = apply_content(image)
    return content_emb

def save_embedding(image_path, output_dir):
    """
    Save the content embedding of an image as a .npy file.
    
    Args:
    - image_path (str): Path to the image file.
    - output_dir (str): Directory to save the .npy file.
    """
    content_emb = process_image(image_path)
    image_name = os.path.basename(image_path)
    npy_filename = os.path.splitext(image_name)[0] + '.npy'
    npy_path = os.path.join(output_dir, npy_filename)
    np.save(npy_path, content_emb)
    print(f"Saved: {npy_path}")

def process_images_in_folder(input_folder, output_folder):
    """
    Process all images in the input folder and save their content embeddings.
    
    Args:
    - input_folder (str): Directory containing the images.
    - output_folder (str): Directory to save the .npy files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            save_embedding(image_path, output_folder)

# Example usage
input_folder = 'data/vimeo_images'
output_folder = 'data/vimeo_conditions/content'
process_images_in_folder(input_folder, output_folder)
