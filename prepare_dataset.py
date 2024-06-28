from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
from utils import *
import glob
import json
import datasets as ds
import os
from PIL import Image

# Path to the images and annotation file
image_paths = glob.glob('/data/maryam.sana/Uni-ControlNet/data/images/*.png')
anno_file = '/data/maryam.sana/Uni-ControlNet/data/data2.txt'

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

file_ids_to_process = [
 'j_032', 'r_036', 'y_038', 'y_044', 'j_002', 'r_039', 'j_060', 'r_019', 'r_048', 'r_017', 
 'y_020', 'y_053', 'j_001-checkpoint', 'j_005', 'r_041', 's_022', 'r_059', 'j_011', 'r_007', 
 'r_045', 'r_008', 'y_058', 'r_054', 'j_050', 'j_026', 'r_029', 'j_020', 'j_003', 'r_046', 
 'j_055', 'j_021', 's_029', 'j_034', 'j_025', 'j_001', 'y_039', 'r_050', 'y_054', 'j_044', 
 'r_005', 'y_023', 'y_048', 's_001', 'y_049', 'r_052', 'y_015', 'j_049', 'y_007', 'y_009', 
 'j_059', 's_026', 'j_043', 'r_060', 'y_047', 's_014', 's_020', 'j_036', 'j_045', 'r_030', 
 'j_013', 'y_011', 'j_023', 'r_006', 'j_046', 'r_049', 'y_059', 'r_021', 'j_002-checkpoint', 
 'y_042', 'y_040', 'y_046', 'r_058', 'y_043'
]

def get_caption(frame):
    prompt = "<image>\nUSER: Give a detailed visual description of this image ?\nASSISTANT:"
    inputs = processor(text=prompt, images=frame, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_length=200)
    caption_llava = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return caption_llava.split('ASSISTANT:')[1].strip()

try:
    with open(anno_file, 'w') as anno:
        for img_path in image_paths:
            try:
                # Get the file ID (assuming it's the filename without the extension)
                file_id = os.path.splitext(os.path.basename(img_path))[0]

                # Process only if the file ID is in the list of files to process
                if file_id in file_ids_to_process:
                    # Load image
                    frame = Image.open(img_path)

                    # Generate caption for the image
                    caption = get_caption(frame)
                    caption = " ".join(caption.split())

                    # Write to the file with tab separation
                    anno.write(f"{file_id}\t{caption}\n")
                    anno.flush()  # Ensure the data is written to the file

                    print(f"Processed: {file_id}")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Annotation file {anno_file} has been created.")
except Exception as e:
    print(f"Failed to create annotation file: {e}")