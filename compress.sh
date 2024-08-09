#!/bin/bash

# Define the directory containing the images
image_folder="data/00096_0576/residues_cs"

# Loop through all PNG images in the folder
for image_path in "$image_folder"/*.png.tfci
do
  # Execute the Python script for each image
  echo $image_path
  python3 tcfi.py decompress "$image_path"
done
