#!/bin/bash

directory="data/vimeo_images"
count=0

for file in "$directory"/*; do
    if [[ $file == *_im2.png ]]; then
        ((count++))
    else
        rm "$file"
    fi
done

echo "Total images remaining: $count"
