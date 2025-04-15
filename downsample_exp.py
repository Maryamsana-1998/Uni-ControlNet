import os
import cv2
from glob import glob
from src.test.test_codec import get_recons_img
from annotator.util import HWC3
from models.util import create_model, load_state_dict

def run_inference_experiments(model, prompt, output_dir="data/inference_results"):
    """
    Run inference with all combinations of frame sizes and canny images
    
    Args:
        model: Your loaded model
        prompt: Text prompt for reconstruction
        output_dir: Where to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define our image options
    frame_options = {
        '512': 'data/UVG_test_data/Beauty/512p/decoded_q4/frame_0000_decoded.png',  # Path to 512px frame image
        '1080': 'data/UVG_test_data/Beauty/1080p/decoded_q4/frame_0000_decoded.png'  # Path to 1080px frame image
    }
    
    # Get all decoded canny images (grid_*_decoded.png)
    canny_images = glob('data/downsample/compressed_flows/grid_*_decoded.png')
    
    # Run all combinations
    for frame_name, frame_path in frame_options.items():
        frame_image = cv2.imread(frame_path)
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
        if frame_image is None:
            print(f"Warning: Could not load frame image at {frame_path}")
            continue
            
        for canny_path in canny_images:
            # Extract grid size from filename
            grid_size = os.path.basename(canny_path).split('_')[1]
            
            # Load canny image
            canny_image = cv2.imread(canny_path)
            canny_image = cv2.cvtColor(canny_image, cv2.COLOR_BGR2RGB)
            if canny_image is None:
                print(f"Warning: Could not load canny image at {canny_path}")
                continue
            
            print(f"Processing: Frame {frame_name} with grid {grid_size}")
            
            # Run reconstruction
            pred_img = get_recons_img(model, prompt, canny_image, frame_image)
            
            # Save result
            output_name = f"recon_frame_{frame_name}_grid_{grid_size}.png"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            
    print("All inference experiments completed!")

# Example usage:
if __name__ == "__main__":
    # Initialize your model here
    model = create_model('configs/uni_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('experiments/exp_perco_lpips2/uni.ckpt', location="cuda"))
    model = model.cuda()
    
    # Your text prompt
    prompt = "A beautiful blonde girl smiling with pink lipstick with black background"  
    
    run_inference_experiments(model, prompt)