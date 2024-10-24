import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset

from .util import *

class UniDataset(Dataset):
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        file_ids, self.annos = read_anno(anno_path)
        self.right_image_paths = [os.path.join(image_dir, 'right', file_id + '.png') for file_id in file_ids]
        self.left_image_paths = [os.path.join(image_dir, 'left', file_id + '.png') for file_id in file_ids]
        self.forward_optical_flow_paths = [os.path.join(condition_root, 'forward', file_id + '.png') for file_id in file_ids]
        self.backward_optical_flow_paths = [os.path.join(condition_root, 'backward', file_id + '.png') for file_id in file_ids]

        # self.global_condition_paths = [os.path.join(global_condition_root, file_id + '.npy') for file_id in file_ids]
       
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
    
    def _load_image(self, image_path):
        """Helper function to load and preprocess an image."""
        image = cv2.imread(image_path)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.resolution, self.resolution))
            image = (image.astype(np.float32) / 127.5) - 1.0
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
        return image
    
    def _load_condition(self, condition_path):
        """Helper function to load and preprocess a condition image."""
        condition = cv2.imread(condition_path)
        try:
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            condition = cv2.resize(condition, (self.resolution, self.resolution))
            condition = condition.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error processing {condition_path}: {e}")
        return condition
    
    def _load_optical_flow(self, flow_path):
        """Helper function to load optical flow data."""
        flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if flow is None:
            print(f"Optical flow not found at {flow_path}")
        return flow

    def _load_global_condition(self, global_file):
        """Helper function to load global conditions."""
        return np.load(global_file)

    def __getitem__(self, index):
        # Item 1: Right image with left image and forward optical flow as conditions
        right_image = self._load_image(self.right_image_paths[index])
        left_condition = self._load_condition(self.left_image_paths[index])
        forward_flow_condition = self._load_optical_flow(self.forward_optical_flow_paths[index])

        # Combine left image and forward optical flow as local conditions for the right image
        right_local_conditions = np.concatenate([left_condition, forward_flow_condition], axis=2)

        # Item 2: Left image with right image and backward optical flow as conditions
        left_image = self._load_image(self.left_image_paths[index])
        right_condition = self._load_condition(self.right_image_paths[index])
        backward_flow_condition = self._load_optical_flow(self.backward_optical_flow_paths[index])

        # Combine right image and backward optical flow as local conditions for the left image
        left_local_conditions = np.concatenate([right_condition, backward_flow_condition], axis=2)
        global_conditions = []
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        # Apply keep and drop logic for both local and global conditions
        if random.random() < self.drop_txt_prob:
            anno = ''  # Drop annotation based on probability
        else:
            anno = self.annos[index]

        # Apply keep and drop logic
        right_local_conditions = keep_and_drop(right_local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        left_local_conditions = keep_and_drop(left_local_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        global_conditions = keep_and_drop(global_conditions, self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)

        # Concatenate local and global conditions if necessary
        if len(right_local_conditions) != 0:
            right_local_conditions = np.concatenate(right_local_conditions, axis=2)
        if len(left_local_conditions) != 0:
            left_local_conditions = np.concatenate(left_local_conditions, axis=2)
        if len(global_conditions) != 0:
            global_conditions = np.concatenate(global_conditions)

        # Return two items with global conditions included
        item1 = dict(jpg=right_image, txt=anno, local_conditions=right_local_conditions, global_conditions=global_conditions)
        item2 = dict(jpg=left_image, txt=anno, local_conditions=left_local_conditions, global_conditions=global_conditions)

        return item1, item2
    
    def __len__(self):
        return len(self.annos)
