import os
import torch
from PIL import Image
from tqdm import tqdm  # Progress bar
from torchvision import transforms
from compressai.zoo import mbt2018
from pathlib import Path

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

# Function to process the dataset directory structure
def process_dataset(root_dir, model, device):
    train_dir = os.path.join(root_dir, 'train')
    
    # Loop through each video directory in train/
    for video in os.listdir(train_dir):
        video_dir = os.path.join(train_dir, video)
        frame_dir = os.path.join(video_dir, 'left')

        # Ensure that the frame directory exists
        if os.path.isdir(frame_dir):
            # Create the output directory inside the video sequence (encoded_frame)
            output_dir = os.path.join(video_dir, 'encoded_left')
            os.makedirs(output_dir, exist_ok=True)
            # Loop through the frame images in the frame directory
            for frame in tqdm(sorted(os.listdir(frame_dir)), desc=f"Processing frames for {video}"):
                frame_path = os.path.join(frame_dir, frame)
                
                if frame.endswith('.png'):
                    # Set the output path for the reconstructed image inside encoded_frame
                    output_path = os.path.join(output_dir, frame)
                    
                    # Process and save the reconstructed image
                    process_and_save_reconstructed_image(frame_path, output_path, model, device)

# Define the main function to initialize the model and process the dataset
def main():
    root_dir = '/data/maryam.sana/datazips/monkaa/'  # Replace with the path to your dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the model (quality level can be changed)
    quality = 8 # Example quality level, you can adjust
    model = mbt2018(quality=quality, pretrained=True).eval().to(device)

    # Process the dataset
    process_dataset(root_dir, model, device)

# Run the main function
if __name__ == '__main__':
    main()
