import einops
import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from test_utils import calculate_metrics
from src.test.test_codec import process_images

# Function to plot original vs predicted images and save the plot
def plot_images(original_images, predictions, save_location, start_index=4, end_index=9, dpi=300):
    fig, axes = plt.subplots(2, end_index - start_index, figsize=(30, 10))

    for i in range(start_index, end_index):
        axes[0, i - start_index].imshow(original_images[i])
        axes[0, i - start_index].set_title(f"Original {i + 1}")
        axes[0, i - start_index].axis('off')
        axes[1, i - start_index].imshow(predictions[i])
        axes[1, i - start_index].set_title(f"Prediction {i + 1}")
        axes[1, i - start_index].axis('off')

    plt.tight_layout()

    # Save the plot as a high-quality image
    plt.savefig(save_location, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved at {save_location}")

    # Optionally close the figure to free up memory
    plt.close(fig)


# Function to calculate PSNR and MS-SSIM
def calculate_image_metrics(original_images, predictions):
    psnr_values = []
    mssim_values = []

    for orig_img, pred_img in zip(original_images, predictions):
        orig_pil = Image.fromarray(orig_img)
        pred_pil = Image.fromarray(pred_img)

        # Calculate PSNR and MS-SSIM
        psnr_value, mssim_value = calculate_metrics(orig_pil, [pred_pil])

        psnr_values.append(psnr_value)
        mssim_values.append(mssim_value)

    # Calculate mean metrics
    mean_psnr = np.mean(psnr_values)
    mean_mssim = np.mean(mssim_values)

    print(f"\nMean PSNR: {mean_psnr:.2f}")
    print(f"Mean MS-SSIM: {mean_mssim:.4f}")

    return mean_psnr, mean_mssim

def main(image_folder, canny_folder, prompt, previous_frame_path, output_folder, num_images=15):
    # Create necessary directories inside the output folder
    pred_folder = os.path.join(output_folder, 'preds')
    residue_folder = os.path.join(output_folder, 'residue')
    plot_save_location = os.path.join(output_folder, 'comparison_plot.png')
    log_file_path = os.path.join(output_folder, 'results_log.txt')
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(residue_folder, exist_ok=True)

    # Get image and canny paths
    image_paths = sorted(glob.glob(f'{image_folder}/*.png'))[:num_images]
    canny_paths = sorted(glob.glob(f'{canny_folder}/*.png'))[:num_images]

    # Process images and get predictions
    original_images, predictions = process_images(image_paths, canny_paths, prompt, previous_frame_path, pred_folder, residue_folder, num_images)

    # Plot images and save the plot in the output folder
    plot_images(original_images, predictions, save_location=plot_save_location)

    # Calculate metrics and print results
    psnr, mssim = calculate_image_metrics(original_images, predictions)

    with open(log_file_path, 'w') as f:
        f.write(f"Processed {image_folder} with previous frame {previous_frame_path}\n")
        f.write(f"PSNR: {psnr}\n")
        f.write(f"MS-SSIM: {mssim}\n")
        f.write(f"Results saved in {output_folder}\n")

    print(f"Results saved to {log_file_path}")

# Example usage for multiple folders and previous frames
if __name__ == "__main__":
    # Define the image folders in the UVG dataset
    image_folders = ['data/UVG/Beauty', 'data/UVG/Bosphorus', 'data/UVG/Jockey']
    canny_folders = ['data/UVG/canny/Beauty', 'data/UVG/canny/Bosphorus', 'data/UVG/canny/Jockey']

    # Define the possible previous frame paths for Bosphorus as an example
    previous_frames = {
        'Bosphorus': [
            'data/UVG/Bosphorus/frame/orig.png',
            'data/UVG/Bosphorus/frame/orig_hi.png',
            'data/UVG/Bosphorus/frame/orig_lo.png'
        ],
        'Beauty': [
            'data/UVG/Beauty/frame/orig.png',
            'data/UVG/Beauty/frame/orig_hi.png',
            'data/UVG/Beauty/frame/orig_lo.png'
        ],
        'Jockey': [
            'data/UVG/Jockey/frame/orig.png',
            'data/UVG/Jockey/frame/orig_hi.png',
            'data/UVG/Jockey/frame/orig_lo.png'
        ]
    }

    # Define a prompt for each video type
    prompts = {
        'Bosphorus': 'a yachet in sea with a flag of turkey waving and a harbour with mountains and trees',
        'Beauty': 'a beautiful landscape with mountains and rivers',
        'Jockey': 'a jockey riding a horse on a sunny day'
    }

    # Iterate over the image folders and process them
    for image_folder, canny_folder in zip(image_folders, canny_folders):
        # Extract the folder name (e.g., 'Bosphorus', 'Beauty', 'Jockey')
        folder_name = os.path.basename(image_folder)
        
        # Loop through each previous frame option for the current folder
        for previous_frame_path in previous_frames[folder_name]:
            output_folder = os.path.join('data/results/vimeo_all', folder_name, os.path.basename(previous_frame_path).split('.')[0])
            print(output_folder)
            # Call the main function
            main(
                image_folder=image_folder,
                canny_folder=canny_folder,
                prompt=prompts[folder_name],
                previous_frame_path=previous_frame_path,
                output_folder=output_folder,
                num_images=15
            )
