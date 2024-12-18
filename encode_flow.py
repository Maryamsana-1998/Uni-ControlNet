import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from compressai.zoo import mbt2018

# Function to process the image and save the reconstructed image
def process_and_save_reconstructed_image(image_path, output_path, model, device): 
    # Load image and convert to a tensor
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)  # Add batch dimension and send to device (CPU/GPU)

    # Run the model in inference mode
    with torch.no_grad():
        out_net = model.forward(x)
        out_net['x_hat'].clamp_(0, 1)  # Clamp values between 0 and 1

    # Convert the reconstructed tensor to a PIL image
    rec_img = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu())

    # Save the reconstructed image
    rec_img.save(output_path)
    print(f"Reconstructed image saved at {output_path}")

# Main script
def process_images(input_folder, output_folder, max_images=20):
    # Load the pre-trained model (quality 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mbt2018(quality=4, pretrained=True).eval().to(device)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of .png image files in the input folder, sorted alphabetically
    image_files = sorted(Path(input_folder).glob("*.png"))[:max_images]

    for image_path in image_files:
        # Use the same filename for the output
        output_path = Path(output_folder) / image_path.name
        process_and_save_reconstructed_image(image_path, output_path, model, device)

# Specify input and output folders
input_folder = "/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/data/UVG/optical_flow/Beauty"
output_folder = "/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/data/UVG/optical_flow/Beauty_reconstructed"

# Run the script
if __name__ == "__main__":
    process_images(input_folder, output_folder)
