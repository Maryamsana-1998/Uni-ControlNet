#!/bin/bash

# Define the directories
input_folder="data/00096_0576_processed"
output_folder="data/00096_0576_processed/compressed_medium"
decomp_folder="data/00096_0576_processed/decompressed_medium"

# Create the output and decompression folders if they don't exist
mkdir -p "$output_folder"
mkdir -p "$decomp_folder"

# Loop through all PNG images in the input folder
for image_path in "$input_folder"/*.png; do
    # Extract the image filename without extension
    image_filename=$(basename "$image_path" .png)

    # Define the output compressed file path
    output_path="$output_folder/$image_filename.tfci"

    # Define the decompressed file path
    decomp_path="$decomp_folder/$image_filename.png"

    # Compress the image
    python3 tcfi.py compress hific-mi "$image_path" "$output_path"

    # Decompress the image
    python3 tcfi.py decompress "$output_path" "$decomp_path"
done

echo "Processing completed!"
