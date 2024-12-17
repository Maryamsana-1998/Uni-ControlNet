import math
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips

# Initialize LPIPS and FID models
lpips_model = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
fid_model = FrechetInceptionDistance(feature=64).to('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    lambda x: x * 255  # Scale to 0-255
])

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def calculate_metrics_batch(original_images, pred_images):
    """
    Calculates PSNR, MS-SSIM, LPIPS, and FID metrics for a batch of image pairs.
    
    Args:
        original_images (list): List of PIL images for the original images.
        pred_images (list): List of PIL images for the predicted images.
    
    Returns:
        dict: Dictionary with metrics for the batch.
    """
    psnr_values, ms_ssim_values, lpips_values = [], [], []
    fid_model.reset()  # Clear any previous FID data

    # Loop over the image pairs
    for original_image, pred_image in zip(original_images, pred_images):
        # Transform images
        original_tensor = transform(original_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        pred_tensor = transform(pred_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate PSNR and MS-SSIM
        psnr_values.append(psnr(original_tensor, pred_tensor).item())
        ms_ssim_values.append(ms_ssim(original_tensor, pred_tensor, data_range=255, size_average=True).item())

        # Calculate LPIPS
        lpips_value = lpips_model(original_tensor / 255.0, pred_tensor / 255.0).item()
        lpips_values.append(lpips_value)

        # Add tensors to FID model
        fid_model.update(original_tensor.to(torch.uint8), real=True)
        fid_model.update(pred_tensor.to(torch.uint8), real=False)

    # Compute FID
    fid_value = fid_model.compute().item()

    return {
        "PSNR": sum(psnr_values) / len(psnr_values),
        "MS-SSIM": sum(ms_ssim_values) / len(ms_ssim_values),
        "LPIPS": sum(lpips_values) / len(lpips_values),
        # "FID": fid_value
    }


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
