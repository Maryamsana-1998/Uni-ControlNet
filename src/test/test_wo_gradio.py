import sys

if "./" not in sys.path:
    sys.path.append("./")
from utils.share import *
import utils.config as config

import einops
import gradio as gr
import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3


from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler


model = create_model("/data/maryam.sana/Uni-ControlNet/configs/laion/uni_v15.yaml").cpu()
model.load_state_dict(load_state_dict("/data/maryam.sana/Uni-ControlNet/checkpoints/laion/uni_v15.ckpt", location="cuda"))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(
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
):

    seed_everything(seed)

    with torch.no_grad():
        original_image = resize_image(HWC3(original_image), image_resolution)

        H, W, C = original_image.shape

        content_emb = np.zeros((768))

        detected_maps = np.concatenate([original_image], axis=2)

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

    return (results, [original_image])


def save_image(image: np.ndarray, name: str):
    assert isinstance(image, np.ndarray) and len(image.shape) == 3
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, image)


def main():
    original_image = cv2.imread("data/000000011.jpg")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    prompt = "Photo of SAM - Zoliborz by <a href=""/members/profile/Vera%20Peres"">Vera Peres</a> <br/>Falafel, tabboulé and vegan mayo <br/> April 4, 2017 - <a href='/contact/abuse/image/89764/244590'>Report</a>"
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
    save_image(results[0], "data/result.jpg")
    save_image(original_image_processed[0], "data/original.jpg")
    print("Done!")


if __name__ == "__main__":
    main()
