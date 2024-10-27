import os
from pathlib import Path
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from PIL import Image


from .util import *

class UniVideoDataset(Dataset):
    """Load a custom video dataset with frames, optical flow, and encoded previous frames, grouped by 5 frames per sequence.

    The dataset should be structured as follows:

    .. code-block::

        - rootdir/
            - train/
                - video1/
                    - frame/
                        - 0001.png
                        - 0002.png
                        - 0003.png
                        - 0004.png
                        - 0005.png
                    - optical_flow/
                        - 0001.flo5 (corresponds to frame 0002.png)
                        - 0002.flo5 (corresponds to frame 0003.png)
                        - 0003.flo5 (corresponds to frame 0004.png)
                        - 0004.flo5 (corresponds to frame 0005.png)
                    - encoded_frame/
                        - 0001.png (corresponds to frame 0002.png)
                        - 0002.png (corresponds to frame 0003.png)
                        - 0003.png (corresponds to frame 0004.png)
                        - 0004.png (corresponds to frame 0005.png)

    Args:
        root (string): root directory of the dataset
        frame_dir (string): name of the directory containing frames
        optical_flow_dir (string): name of the optical flow directory
        encoded_frame_dir (string): name of the directory for encoded previous frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'valid')
    """

    def __init__(self, root, frame_dir, optical_flow_dir, encoded_frame_dir,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob, split="train"):
        # Define the path to the root directory
        self.root = Path(root) / split
        self.frame_dir = frame_dir
        self.optical_flow_dir = optical_flow_dir
        self.encoded_frame_dir = encoded_frame_dir
        self.samples = self._load_samples()
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.global_type_list = global_type_list

    def _load_samples(self):
        """Load the samples from the directory structure, grouped into sets of 5 frames, skipping the first frame for each group."""
        samples = []
        for video_folder in self.root.iterdir():
            frame_folder = video_folder / self.frame_dir
            optical_flow_folder = video_folder / self.optical_flow_dir
            encoded_frame_folder = video_folder / self.encoded_frame_dir
            
            # Ensure all directories exist
            if frame_folder.exists() and optical_flow_folder.exists() and encoded_frame_folder.exists():
                frame_files = sorted(frame_folder.glob("*.png"))
                optical_flow_files = sorted(optical_flow_folder.glob("*.png"))
                encoded_frame_files = sorted(encoded_frame_folder.glob("*.png"))

                # Ensure we have enough frames and conditions to form groups of 5
                for i in range(1, len(frame_files) - 3):  # -4 to ensure a group of 5 frames
                    frame_group = frame_files[i:i + 3]
                    optical_flow_group = optical_flow_files[i-1:i + 2]  # 4 optical flows correspond to frames 2-5
                    encoded_frame_group = encoded_frame_files[i-1:i + 2]  # 4 encoded frames correspond to frames 2-5
                    
                    # Only add samples if we have exactly 5 frames and 4 corresponding conditions
                    if len(frame_group) ==3 and len(optical_flow_group) == 3 and len(encoded_frame_group) == 3:
                        samples.append((frame_group, optical_flow_group, encoded_frame_group))

        return samples
 

    def load_image(self, frame_path):
        """Load and process a frame (image)."""
        # print(frame_path)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (self.resolution, self.resolution))  # Resize to target resolution
        image = (image.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1] range    
        return image  # Return None if an exception occurred
        
    def load_local_condition(self, optical_flow_path, encoded_frame_path):
        """Load the optical flow and the encoded previous frame."""
        
        # List of files to process (optical flow and encoded frame)
        local_files = [optical_flow_path, encoded_frame_path]
        
        # This will hold the processed conditions
        local_conditions = []
        
        for local_file in local_files:
            condition = cv2.imread(local_file)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0  # Normalize to [0, 1]
            local_conditions.append(condition)
            # except Exception as e:
            #     print(f"Error processing file {local_file}: {e}")
    
        # Apply keep and drop logic (presumably user-defined function)
        local_conditions = keep_and_drop(local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
    
        # Concatenate the conditions along the channel axis if not empty
        if len(local_conditions) != 0:
            local_conditions = np.concatenate(local_conditions, axis=2)
    
        return local_conditions

    def load_global_condition(self,index,global_type_list):
        global_conditions = []
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)
        return global_conditions 
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {
                "jpg": List of `PIL.Image.Image` or transformed `PIL.Image.Image` (group of 5),
                "txt": List of text annotations or empty strings,
                "local_conditions": List of [optical_flow, encoded_frame] for frames 2-5,
                "global_conditions": List of global conditions for the corresponding frames
            }
        """
        # Unpack the sample tuple
        frame_group, optical_flow_group, encoded_frame_group = self.samples[index]

        # Initialize empty lists to store data for all frames in this sample
        jpg = []
        txt = []  # If you have text data associated with frames, you can modify this
        local_conditions = []
        global_conditions = []

        # Iterate through frames and corresponding conditions (starting from frame 2)
        for i, (frame_path, opt_flow, enc_frame) in enumerate(zip(frame_group, optical_flow_group, encoded_frame_group)):
            # Load the image for the current frame
            image = self.load_image(frame_path)
            jpg.append(image)  # Append to the jpg list

            # Add empty strings or actual annotations to txt if available
            # txt.append('predict next image')  # Placeholder for annotations

            # Load local conditions (optical flow and encoded frame)
            local_condition = self.load_local_condition(opt_flow, enc_frame)
            local_conditions.append(local_condition)

            # Load global conditions (assumed that global conditions are per frame)
            global_condition = self.load_global_condition(index, self.global_type_list)
            global_conditions.append(global_condition)

        jpg = np.stack(jpg, axis=0)  # Convert list of images to numpy array (shape will be [5, H, W, C])
        # print(jpg.shape)
        # txt = np.array(txt)  # Convert list of text placeholders to numpy array
        local_conditions = np.stack(local_conditions, axis=0)  # Convert list of local conditions to numpy array
        global_conditions = np.stack(global_conditions, axis=0)  # Convert list of global conditions to numpy array

        # Return a single dictionary containing jpg, txt, local conditions, and global conditions
        return {
            'jpg': jpg,  # List of all images (frames)
            'txt': ['predict next image','predict next image','predict next image'],  # Placeholder for text annotations (can be modified)
            'local_conditions': local_conditions,  # List of local conditions for each frame
            'global_conditions': global_conditions  # List of global conditions for each frame
        }


    def __len__(self):
        """Returns the total number of samples (groups of 5 frames with corresponding conditions)"""
        return len(self.samples)
