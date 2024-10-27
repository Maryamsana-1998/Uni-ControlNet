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
        """Load the samples from the directory structure as individual frames and conditions, not grouped."""
        samples = []
        for video_folder in self.root.iterdir():
            try:
                frame_folder = video_folder / self.frame_dir
                optical_flow_folder = video_folder / self.optical_flow_dir
                encoded_frame_folder = video_folder / self.encoded_frame_dir

                # Ensure all directories exist
                if frame_folder.exists() and optical_flow_folder.exists() and encoded_frame_folder.exists():
                    frame_files = sorted(frame_folder.glob("*.png"))
                    optical_flow_files = sorted(optical_flow_folder.glob("*.png"))
                    encoded_frame_files = sorted(encoded_frame_folder.glob("*.png"))
                    # print(len(optical_flow_files), len(frame_files))

                    # Ensure we have enough frames and corresponding conditions
                    for i in range(1, len(frame_files)):
                        frame = frame_files[i]
                        optical_flow = optical_flow_files[i-1]  # Optical flow corresponds to the previous frame
                        encoded_frame = encoded_frame_files[i - 1]  # Encoded frame corresponds to the previous frame
                        # print('missing:',video_folder,i)

                        # Only add samples if all corresponding files exist
                        if frame and optical_flow and encoded_frame:
                            samples.append((frame, optical_flow, encoded_frame))
            except:
                print(video_folder)

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
                "jpg": np.ndarray of shape [H, W, C],
                "txt": Single text annotation or empty string,
                "local_conditions": np.ndarray for the optical flow and encoded frame conditions,
                "global_conditions": np.ndarray for the global conditions for this frame
            }
        """
        # Unpack the sample tuple - each entry corresponds to one frame and its conditions
        frame_path, opt_flow, enc_frame = self.samples[index]

        # Load the image for the current frame
        jpg = self.load_image(frame_path)  # Shape [H, W, C]

        # Load the local conditions (optical flow and encoded frame)
        local_conditions = self.load_local_condition(opt_flow, enc_frame)

        # Load the global conditions for this frame
        global_conditions = self.load_global_condition(index, self.global_type_list)

        # Set text annotation (for example, you can modify it as needed)
        txt = 'predict next image'  # Placeholder for text annotations

        # Return a single dictionary containing the image, txt, local conditions, and global conditions
        return {
            'jpg': jpg,  # Single image as a numpy array [H, W, C]
            'txt': txt,  # Text annotation for this frame
            'local_conditions': local_conditions,  # Local conditions as numpy array
            'global_conditions': global_conditions  # Global conditions as numpy array
        }


    def __len__(self):
        """Returns the total number of samples (groups of 5 frames with corresponding conditions)"""
        return len(self.samples)
