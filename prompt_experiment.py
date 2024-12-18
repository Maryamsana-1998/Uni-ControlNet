import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from src.test.test_codec import process_images

# Define paths
base_data_path = "/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/data/UVG"
ckpt_path = "/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/checkpoints/vimeo_8/uni_v8.ckpt"
cfg_path = "configs/uni_v15.yaml"
pred_folder = "exp/"

# Video-specific prompts
video_details = {
    "Beauty": {
        "prompt": "A beautiful blonde girl with pink lipstick",
        "path": os.path.join(base_data_path, "Beauty")
    },
    "Jockey": {
        "prompt": "The image features a man riding a brown horse, galloping through a grassy field. The man is wearing a yellow shirt and is skillfully guiding the horse.",
        "path": os.path.join(base_data_path, "Jockey")
    },
    "Bosphorus": {
        "prompt": "The image features a man and a woman sitting together on a boat in the water. They are both wearing ties, suggesting a formal or semi-formal occasion.",
        "path": os.path.join(base_data_path, "Bosphorus")
    }
}

# Define canny and previous frame subfolders
canny_subfolder = "optical_flow"
previous_frame_subfolder = "quality_8"

# Number of images to process
num_images = 2

def plot_comparison(original_image, prediction_with_prompt, prediction_without_prompt, save_path):
    """
    Plots original image, prediction with prompt, and prediction without prompt side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # With Prompt
    axes[1].imshow(prediction_with_prompt)
    axes[1].set_title("With Prompt")
    axes[1].axis('off')

    # Without Prompt
    axes[2].imshow(prediction_without_prompt)
    axes[2].set_title("Without Prompt")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Comparison saved at {save_path}")
    plt.close(fig)

# Loop through each video and perform inference
for video_name, details in video_details.items():
    print(f"\nProcessing video: {video_name}")

    original_folder = details["path"]
    canny_folder = os.path.join(base_data_path, canny_subfolder, video_name)
    previous_frame_folder = os.path.join(original_folder, previous_frame_subfolder)

    image_paths = sorted(glob.glob(os.path.join(original_folder, "*.png")))
    canny_paths = sorted(glob.glob(os.path.join(canny_folder, "*.png")))
    previous_frames_paths = sorted(glob.glob(os.path.join(previous_frame_folder, "*.png")))

    # Process with prompt
    predictions_with_prompt = process_images(
        config_path=cfg_path,
        ckpt_path=ckpt_path,
        image_paths=image_paths,
        canny_paths=canny_paths,
        prompt=details["prompt"],
        previous_frames_paths=previous_frames_paths,
        pred_folder=os.path.join(pred_folder, video_name, "with_prompt"),
        num_images=num_images
    )[1]

    # Process without prompt
    predictions_without_prompt = process_images(
        config_path=cfg_path,
        ckpt_path=ckpt_path,
        image_paths=image_paths,
        canny_paths=canny_paths,
        prompt="",  # Empty prompt
        previous_frames_paths=previous_frames_paths,
        pred_folder=os.path.join(pred_folder, video_name, "without_prompt"),
        num_images=num_images
    )[1]

    # Load original images
    original_images = [Image.open(path).convert("RGB") for path in image_paths[:num_images]]
    # print(original_images,pred)
    # Plot and save comparison
    save_path = os.path.join(pred_folder, video_name, f"{video_name}_comparison.png")
    plot_comparison(original_images[0], predictions_with_prompt[0], predictions_without_prompt[0], save_path)
