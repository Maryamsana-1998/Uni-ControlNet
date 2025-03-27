import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# Importing specific utilities
from src.test.test_codec import process
from utils.share import *
import utils.config as config
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler


ckpt_path= "./exp_perco_lpips2/uni.ckpt"
config_path ="./configs/vimeo_lpips/uni_v15.yaml"
prompt = "A beautiful blonde girl with pink lipstick"
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(ckpt_path, location="cuda"))
model = model.cuda()
a_prompt = "best quality, extremely detailed"
n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
num_samples = 1
image_resolution = 512
ddim_steps = 50
strength = 1
scale = 7.5
seed = 42
eta = 0.0
global_strength = 1

image_paths = sorted(glob.glob('../vimeo_unicontrol/Uni-ControlNet/data/UVG/images/original/Jockey/*.png'))[0:20]
optical_flows = sorted(glob.glob('../vimeo_unicontrol/Uni-ControlNet/data/UVG/optical_flow/decoded/Jockey5/*.png'))
encoded_frames = sorted(glob.glob('../vimeo_unicontrol/Uni-ControlNet/data/UVG/images/decoded/Jockey/quality_4/*.png'))[0:20]  # Load encoded frame
# print(image_paths,optical_flows,encoded_frames)
prompt = "A beautiful blonde girl with pink lipstick"


pred_folder = "test_codec/revised_Jockey/"
os.makedirs(pred_folder, exist_ok=True)

# Load images
original_images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]
optical_flow_images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in optical_flows]
encoded_images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in encoded_frames]

# List to store predictions
predictions = []

# Process images
for i in range(len(image_paths)):
    original_image = original_images[i]
    if i % 5 == 0:
        pred_img = encoded_images[i]
        print('use base frame' , os.path.basename(encoded_frames[i]))
    else: 
        base_frame_idx = i//5 
        print(f'For generating {i+1}, we have base frame: {os.path.basename(encoded_frames[base_frame_idx*5])} ',
              f'and optical flow between base and frame: {os.path.basename(optical_flows[i-1-base_frame_idx])}')

        frame_image = encoded_images[base_frame_idx*5]
        canny_image = optical_flow_images[i-1-base_frame_idx] ##op image
        #Use the `process` function
        pred = process(
            model,
            canny_image,
            frame_image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            ddim_steps,
            strength,
            scale,
            seed,
            eta,
            global_strength,
        )
        pred_img = pred[0][0]
    
        # Ensure the images are of the same size
        if original_image.shape != pred_img.shape:
            pred_img = cv2.resize(pred_img, (original_image.shape[1], original_image.shape[0]))
    
    # Save the prediction for the next frame
    predictions.append(pred_img)

    # Save prediction image to disk
    pred_image_path = os.path.join(pred_folder, f"im{i + 1}_pred.png")
    cv2.imwrite(pred_image_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
    print(f"Saved prediction image: {pred_image_path}")
