from transformers import AutoProcessor, LlavaForConditionalGeneration
import numpy as np
import glob
import os
from PIL import Image
from tqdm import tqdm

# Path to the images and annotation file
image_paths = glob.glob('/data/maryam.sana/Uni-ControlNet/data/vimeo_data/vimeo_images/*.png')
anno_file = '/data/maryam.sana/Uni-ControlNet/data/vimeo_data/vimeo_data.txt'

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def get_caption(frame):
    prompt = "<image>\nUSER: Give a detailed visual description of this image ?\nASSISTANT:"
    inputs = processor(text=prompt, images=frame, return_tensors="pt")
    generate_ids = model.generate(**inputs, max_length=200)
    caption_llava = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return caption_llava.split('ASSISTANT:')[1].strip()

# Read existing annotations
existing_annotations = set()
if os.path.exists(anno_file):
    with open(anno_file, 'r') as anno:
        for line in anno:
            file_id, caption = line.strip().split('\t')
            existing_annotations.add(file_id)

# Filter out already processed images
image_paths = [img_path for img_path in image_paths if os.path.splitext(os.path.basename(img_path))[0] not in existing_annotations]

print(f"Total images to process: {len(image_paths)}")

try:
    with open(anno_file, 'a') as anno:
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Get the file ID (assuming it's the filename without the extension)
                file_id = os.path.splitext(os.path.basename(img_path))[0]

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

    print(f"Annotation file {anno_file} has been updated.")
except Exception as e:
    print(f"Failed to update annotation file: {e}")
