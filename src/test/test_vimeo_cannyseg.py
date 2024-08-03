import sys

if "./" not in sys.path:
    sys.path.append("./")
from utils.share import *
import utils.config as config

import einops
import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3

from annotator.canny import CannyDetector
from annotator.uniformer import UniformerDetector
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler


model = create_model("/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/configs/vimeo/uni_v15.yaml").cpu()
model.load_state_dict(load_state_dict("/data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/checkpoints/vimeo/uni_v15.ckpt", location="cuda"))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
apply_canny = CannyDetector()
apply_seg = UniformerDetector()

def process(
    canny_image,
    seg_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    strength,
    scale,
    seed,
    eta,
    global_strength,
    low_threshold=100, high_threshold=200, 
):

    seed_everything(seed)

    with torch.no_grad():
        # original_image = resize_image(HWC3(original_image), image_resolution)

        # H, W, C = original_image.shape
        W,H = image_resolution,image_resolution

        canny_image = cv2.resize(canny_image, (W, H))
        canny_detected_map = HWC3(apply_canny(HWC3(canny_image), low_threshold, high_threshold))

        seg_image = cv2.resize(seg_image, (W, H))
        seg_detected_map, _ = apply_seg(HWC3(seg_image))
        seg_detected_map = HWC3(seg_detected_map)

        content_emb = np.zeros((768))

        detected_maps = np.concatenate([canny_detected_map,seg_detected_map], axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, "b h w c -> b c h w").clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack(
            [global_control for _ in range(num_samples)], dim=0
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {
            "local_control": [local_control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
            "global_control": [global_control],
        }
        un_cond = {
            "local_control": [uc_local_control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
            "global_control": [uc_global_control],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=True,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            global_strength=global_strength,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        results = [x_samples[i] for i in range(num_samples)]

    return (results, [canny_detected_map,seg_detected_map])


def save_image(image: np.ndarray, name: str):
    assert isinstance(image, np.ndarray) and len(image.shape) == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, image)


def main():
    original_image = cv2.imread("data/00001_0008_im2.png")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    prompt = "The image features a man wearing a hat, standing in front of a tree. He appears to be looking at the camera, possibly posing for a picture. The man is wearing a blue shirt, and there is a tie visible on him. The scene is set in a garden, with the tree providing a natural backdrop."
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    num_samples = 1
    image_resolution = 512
    ddim_steps = 50
    strength = 1
    scale = 7.5
    seed = 42
    eta = 0.0
    global_strength = 1

    args = [
        original_image,
        original_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        strength,
        scale,
        seed,
        eta,
        global_strength,
    ]
    results, original_image_processed = process(*args)
    save_image(results[0], "data/result_cs.jpg")
    save_image(original_image_processed[0], "data/original_cs.jpg")
    print("Done!")


if __name__ == "__main__":
    main()
