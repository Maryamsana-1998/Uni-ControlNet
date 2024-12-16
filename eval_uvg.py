import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
from src.test.test_codec import process_images
from test_utils import calculate_metrics_batch

# Define dataset directories
original_root = "/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/data/UVG/"
pred_root = "experiment_155218/predictions/"

# Videos to process
videos = ["Beauty", "Bosphorus", "Jockey"]

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
    plt.savefig(save_location, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved at {save_location}")
    plt.close(fig)



# Initialize a dictionary to store metrics for each video
all_metrics = {}

# Loop through each video
for video in videos:
    print(f"\nProcessing video: {video}")

    # Define folders for this video (For demonstration, assuming same directory structure as your single-video example)
    original_folder = os.path.join(original_root, video)
    canny_folder = os.path.join(original_root, "optical_flow", video)
    previous_frame_folder = os.path.join(original_root, video, "quality_8")
    pred_folder = os.path.join(pred_root, video)

    # Retrieve image paths for inference
    image_paths = sorted(glob.glob(os.path.join(original_folder, "*.png")))
    canny_paths = sorted(glob.glob(os.path.join(canny_folder, "*.png")))
    previous_frames_paths = sorted(glob.glob(os.path.join(previous_frame_folder, "*.png")))

    # Define the prompt if any (empty in this case)
    prompt = ''

    # Number of images you want to process (adjust as needed)
    num_images = 10

    # Ensure prediction directory exists
    os.makedirs(pred_folder, exist_ok=True)

    # Run inference for this video
    original_images, predictions = process_images(
        image_paths=image_paths,
        canny_paths=canny_paths,
        prompt=prompt,
        previous_frames_paths=previous_frames_paths,
        pred_folder=pred_folder,
        num_images=num_images
    )

    # Define the save location for the plot (optional)
    plot_save_location = os.path.join(pred_folder, "original_vs_predicted.png")

    # Plot the images and save the plot (optional visualization)
    if len(original_images) > 5 and len(predictions) > 5:
        plot_images(
            original_images=original_images,
            predictions=predictions,
            save_location=plot_save_location,
            start_index=2,
            end_index=5,
            dpi=300
        )

    # Evaluation: compute metrics from im2 to im10
    # Collect original and predicted images for metric calculation
    original_eval_images = []
    pred_eval_images = []

    for i in range(2, 10):
        original_path = os.path.join(original_root, video, f"im{i:05d}.png")  # im00002.png
        pred_path = os.path.join(pred_root, video, f"im{i}_pred.png")       # im2_pred.png

        if os.path.exists(original_path) and os.path.exists(pred_path):
            original_eval_images.append(Image.open(original_path).convert("RGB"))
            pred_eval_images.append(Image.open(pred_path).convert("RGB"))
        else:
            print(f"Warning: Missing image for {video} frame {i}")

    # Calculate metrics for this video
    if original_eval_images and pred_eval_images:
        metrics = calculate_metrics_batch(original_eval_images, pred_eval_images)
        all_metrics[video] = metrics
        print(f"Metrics for {video}:", metrics)
    else:
        print(f"No images found or incomplete data for video {video}. Skipping metrics.")


# Print summary of all metrics
print("\nFinal Metrics Summary for All Videos:")
for video, metrics in all_metrics.items():
    print(f"{video} Metrics: {metrics}")

# Save the metrics to a JSON file
metrics_json_path = "all_videos_metrics.json"
with open(metrics_json_path, "w") as f:
    json.dump(all_metrics, f, indent=4)

print(f"\nAll metrics saved to {metrics_json_path}")