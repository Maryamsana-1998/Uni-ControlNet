import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import cv2

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
                for i in range(1, len(frame_files) - 4):  # -4 to ensure a group of 5 frames
                    frame_group = frame_files[i:i + 5]
                    optical_flow_group = optical_flow_files[i:i + 5]  # 4 optical flows correspond to frames 2-5
                    encoded_frame_group = encoded_frame_files[i-1:i + 4]  # 4 encoded frames correspond to frames 2-5
                    
                    # Only add samples if we have exactly 5 frames and 4 corresponding conditions
                    if len(frame_group) == 5 and len(optical_flow_group) == 5 and len(encoded_frame_group) == 5:
                        samples.append((frame_group, optical_flow_group, encoded_frame_group))

        return samples
 
    def load_image(self, frame_path):
        """Load and process a frame (image)."""
        image = None  # Initialize image as None
    
        try:
            # Load the image
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, (self.resolution, self.resolution))  # Resize to target resolution
            image = (image.astype(np.float32) / 127.5) - 1.0  # Normalize to [-1, 1] range
    
        except Exception as e:
            print(f"Error processing image {frame_path}: {e}")
        
        return image  # Return None if an exception occurred
        
    def load_local_condition(self, optical_flow_path, encoded_frame_path):
        """Load the optical flow and the encoded previous frame."""
        
        # List of files to process (optical flow and encoded frame)
        local_files = [optical_flow_path, encoded_frame_path]
        
        # This will hold the processed conditions
        local_conditions = []
        
        for local_file in local_files:
            condition = cv2.imread(local_file)
            try:    
                # Convert BGR to RGB, resize, and normalize
                condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
                condition = cv2.resize(condition, (self.resolution, self.resolution))
                condition = condition.astype(np.float32) / 255.0  # Normalize to [0, 1]
                local_conditions.append(condition)
            except Exception as e:
                print(f"Error processing file {local_file}: {e}")
    
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
                "image": List of `PIL.Image.Image` or transformed `PIL.Image.Image` (group of 5),
                "local_condition": List of [optical_flow, encoded_frame] for frames 2-5
            }
        """
        # Unpack the sample tuple
        frame_group, optical_flow_group, encoded_frame_group = self.samples[index]

        # Initialize an empty list to store the dictionaries
        output_dicts = []

        # Iterate through frames and conditions (starting from frame 2 as per your logic)
        for i, (frame_path, opt_flow, enc_frame) in enumerate(zip(frame_group, optical_flow_group, encoded_frame_group)):
            # Load the image for the current frame
            image = self.load_image(frame_path)

            # Load local conditions (optical flow and encoded frame)
            local_condition = self.load_local_condition(opt_flow, enc_frame)

            # Load global conditions
            global_condition = self.load_global_condition(index, self.global_type_list)

            # Create a dictionary for the current frame
            frame_dict = {
                'jpg': image,  # Processed frame image
                'txt': '',     # Annotation placeholder, as per your original code
                'local_conditions': local_condition,  # Corresponding local condition
                'global_conditions': global_condition  # Corresponding global condition
            }

            # Append the dictionary to the output list
            output_dicts.append(frame_dict)

        # Return the array of dictionaries
        return output_dicts

    def __len__(self):
        """Returns the total number of samples (groups of 5 frames with corresponding conditions)"""
        return len(self.samples)
