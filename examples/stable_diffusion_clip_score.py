"""
ref: https://huggingface.co/docs/diffusers/conceptual/evaluation
"""

from functools import partial
import sys
import os
import copy
from pathlib import Path
import csv

import torch
from torchmetrics.functional.multimodal import clip_score
from diffusers import DiffusionPipeline

current_dir = os.path.join(Path().resolve())
sys.path.append(str(current_dir) + "/../")

from models.converter import analog_convert

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]


model_id = "CompVis/ldm-text2im-large-256"
model = DiffusionPipeline.from_pretrained(model_id).to("cuda")
images = model(prompts, num_images_per_prompt=100, output_type="np").images

sd_clip_score = calculate_clip_score(images, prompts)
print(f"Original CLIP score: {sd_clip_score}")

targets = ["to_k", "to_q", "to_v"]

bits = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
stds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

with open("clip-score.csv", "w") as f:
    print(f"Original = {sd_clip_score}", file=f)

    writer = csv.writer(f)
    writer.writerow([""] + stds)

    for bit in bits:
        scores = []
        for std in stds:
            model_analog = copy.deepcopy(model)
            analog_convert(
                model_analog.unet,
                in_bit=bit,
                out_bit=bit,
                w_bit=bit,
                std=std,
                targets=targets,  # if specify None, all layers would be converted to analog
                convert_linear=True,
                convert_conv2d=True,
                type="affine",
                verbose=False,
            )
            model_analog.to("cuda")
            images = model_analog(prompts, num_images_per_prompt=100, output_type="np").images

            sd_clip_score = calculate_clip_score(images, prompts)
            scores.append(sd_clip_score)
            print(f"{bit}bit-{std}std CLIP score: {sd_clip_score}")

        writer.writerow([bit] + scores)
