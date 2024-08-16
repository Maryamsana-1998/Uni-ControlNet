#!/bin/bash

# Define the directories
input_folder="data/00096_0576_processed/mi/decompressed"
canny_folder="$input_folder/canny"
seg_folder="$input_folder/seg"

# Create the canny and seg folders if they don't exist
mkdir -p "$canny_folder"
mkdir -p "$seg_folder"

# Move canny images to the canny folder
mv "$input_folder"/*_canny.png "$canny_folder"

# Move seg images to the seg folder
mv "$input_folder"/*_seg.png "$seg_folder"

echo "Images have been organized into 'canny' and 'seg' folders."
