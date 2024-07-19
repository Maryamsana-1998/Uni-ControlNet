# import os
# import random
# import time
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# import matplotlib.pyplot as plt
# import torch

# # Initialize the model and processor
# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="cuda")

# # Ensure the model is on the same device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Function to generate caption for an image
# def generate_caption(image_path):
#     raw_image = Image.open(image_path).convert('RGB')
#     question = "Describe this image in detail."
#     inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)
    
#     # Debugging: Check the shape and device of the inputs
#     print(f"Inputs shape: {inputs['input_ids'].shape}, device: {inputs['input_ids'].device}")
    
#     generate_ids = model.generate(**inputs)
#     caption = processor.decode(generate_ids[0], skip_special_tokens=True).strip()
    
#     # Debugging: Check the generated IDs and caption
#     print(f"Generated IDs: {generate_ids}")
#     print(f"Caption: {caption}")
    
#     return caption

# # Directory containing images
# image_dir = 'data/uvg_data/images/'

# # Get a list of all images in the directory
# all_images = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

# # Randomly select 10 images
# random_images = random.sample(all_images, 10)

# # Generate captions and measure inference time
# captions = []
# start_time = time.time()

# for img_path in random_images:
#     caption = generate_caption(img_path)
#     captions.append((img_path, caption))

# end_time = time.time()
# inference_time = end_time - start_time

# # Plot images with captions
# fig, axs = plt.subplots(2, 5, figsize=(20, 10))
# axs = axs.flatten()

# for ax, (img_path, caption) in zip(axs, captions):
#     img = Image.open(img_path)
#     ax.imshow(img)
#     ax.set_title(caption, fontsize=10)
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig('image_captions.png')
# plt.show()

# print(f"Total inference time for 10 images: {inference_time:.2f} seconds")
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT
# question = "Describe objects and colors in this image"
# #     inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)
image = Image.open('/data/maryam.sana/Uni-ControlNet/data/test_data/j_017.png')
prompt = "Question: describe objects in image? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
