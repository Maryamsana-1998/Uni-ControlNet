#!/bin/bash

# Define the directory containing the images
image_folder="data/00096_0576/cannyseg/mi/residues/"

# Loop through all PNG images in the folder
for image_path in "$image_folder"/*.png; do
  # Define the output compressed file path by changing the extension to .tfci
  compressed_path="${image_path}.tfci"

  # Compress the image
  python3 tcfi.py compress hific-lo "$image_path" "$compressed_path"

  # Decompress the image (output will be saved back as .png)
  python3 tcfi.py decompress "$compressed_path" 

done

echo "All images have been processed successfully!"
