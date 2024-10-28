import math
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Transformation to resize and convert images to tensor
psnr_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    lambda x: x * 255
])

def resize_image(image, size=(512, 512)):
    return image.resize(size, Image.LANCZOS)

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

# Function to calculate PSNR and MS-SSIM for an image against a list of images
def calculate_metrics(reference_image, pil_images):
    # Transform the reference image
    ref_tensor = psnr_transform(reference_image)

    psnr_values = []
    ms_ssim_values = []

    for image in pil_images:
        # Transform the current image
        img_tensor = psnr_transform(image)

        # Calculate PSNR
        psnr_value = psnr(ref_tensor, img_tensor)
        psnr_values.append(psnr_value)

        # Calculate MS-SSIM
        ms_ssim_value = ms_ssim(ref_tensor.unsqueeze(0), img_tensor.unsqueeze(0), data_range=255, size_average=True).item()
        ms_ssim_values.append(ms_ssim_value)

    return psnr_values, ms_ssim_values


# Function to visualize results
def visualize_results(input_image, detected_maps, predicted_images, condition_name):
    num_samples = len(predicted_images)
    num_conditions = len(detected_maps)
    
    fig, axs = plt.subplots(2, max(num_samples, num_conditions + 1), figsize=(20, 10))
    
    # Display input image (ground truth)
    axs[0, 0].imshow(input_image)
    axs[0, 0].set_title('Ground Truth')
    axs[0, 0].axis('off')
    
    # Display detected maps (conditions)
    for i in range(num_conditions):
        axs[0, i + 1].imshow(detected_maps[i])
        axs[0, i + 1].set_title(f'Condition {i + 1}')
        axs[0, i + 1].axis('off')
    
    # Display predicted images
    for i in range(num_samples):
        axs[1, i].imshow(predicted_images[i])
        axs[1, i].set_title(f'Predicted Image {i + 1}')
        axs[1, i].axis('off')
    
    # Hide any unused subplots
    for i in range(num_conditions + 1, max(num_samples, num_conditions + 1)):
        axs[0, i].axis('off')
    for i in range(num_samples, max(num_samples, num_conditions + 1)):
        axs[1, i].axis('off')
    
    plt.show()
