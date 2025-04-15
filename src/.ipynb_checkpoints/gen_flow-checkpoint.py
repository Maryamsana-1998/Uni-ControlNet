import os
import shutil
import subprocess
import matplotlib.pyplot as plt
import flowiz as fz
from tqdm import tqdm
import sys
sys.stdout.reconfigure(line_buffering=True)

def compute_optical_flow_for_folder(folder_path, gpu_id=0):
    r1_path = os.path.join(folder_path, "r1.png")
    flow_dir = os.path.join(folder_path, "Flow")
    os.makedirs(flow_dir, exist_ok=True)

    if not os.path.exists(r1_path) :
        print(f"Missing r1.png or depth folder in {folder_path}")
        return

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(('.png', '.jpg')) and f not in ['r1.png', 'r2.png']
    ])

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        flo_name = os.path.splitext(img_name)[0] + ".flo"
        png_name = os.path.splitext(img_name)[0] + ".png"

        flo_output_path = os.path.join(flow_dir, flo_name)
        png_output_path = os.path.join(flow_dir, png_name)
        # print("*****************************************")
        # print(' image and save paths ', img_path, png_output_path)    
         
        try:
            command = [
                "python3", "run.py",
                "--model", "sintel-final",
                "--one", r1_path,
                "--two", img_path,
                "--out", flo_output_path
            ]

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            subprocess.run(command, env=env, check=True)

            # Convert and save flow visualization
            flow_img = fz.convert_from_file(flo_output_path)
            plt.imsave(png_output_path, flow_img)
            print("done", png_output_path, flush=True)

        except Exception as e:
            print(f"Error processing {img_name} in {folder_path}: {e}")

def process_all_folders(base_dir, gpu_id=0):
    for root, dirs, files in tqdm(os.walk(base_dir)):
        if "r1.png" in files and "depth" in dirs:
            compute_optical_flow_for_folder(root, gpu_id=gpu_id)
            # break

# Usage
base_vimeo_dir = "/data2/local_datasets/vimeo_septuplet/sequences"
process_all_folders(base_vimeo_dir, gpu_id=0)
