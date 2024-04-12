import sys
import os
from zipfile import ZipFile
from pathlib import Path
import requests
import csv
import copy

import torch
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as F
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance

current_dir = os.path.join(Path().resolve())
sys.path.append(str(current_dir) + "/../")

from models.converter import analog_convert


seed = 0
generator = torch.manual_seed(seed)


def download(url, local_filepath):
    r = requests.get(url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)
    return local_filepath


dummy_dataset_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/sample-imagenet-images.zip"
local_filepath = download(dummy_dataset_url, dummy_dataset_url.split("/")[-1])

with ZipFile(local_filepath, "r") as zipper:
    zipper.extractall(".")


dataset_path = "sample-imagenet-images"
image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))


real_images = torch.cat([preprocess_image(image) for image in real_images])

dit_pipeline = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
dit_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(dit_pipeline.scheduler.config)
dit_pipeline = dit_pipeline.to("cuda")

words = [
    "cassette player",
    "chainsaw",
    "chainsaw",
    "church",
    "gas pump",
    "gas pump",
    "gas pump",
    "parachute",
    "parachute",
    "tench",
]

class_ids = dit_pipeline.get_label_ids(words)
fake_images = dit_pipeline(class_labels=class_ids, generator=generator, output_type="np").images
fake_images = torch.tensor(fake_images).permute(0, 3, 1, 2)

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)
score = fid.compute()
print(f"Original FID: {float(score)}")

targets = ["to_k", "to_q", "to_v"]

bits = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
stds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

with open("fid-score.csv", "w") as f:
    print(f"Original = {float(score)}", file=f)

    writer = csv.writer(f)
    writer.writerow([""] + stds)

    for bit in bits:
        scores = []
        for std in stds:
            dit_pipeline_analog = copy.deepcopy(dit_pipeline)
            analog_convert(
                dit_pipeline_analog.transformer,
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
            dit_pipeline_analog.to("cuda")
            print(dit_pipeline_analog.transformer)
            fake_images = dit_pipeline_analog(class_labels=class_ids, generator=generator, output_type="np").images
            fake_images = torch.tensor(fake_images).permute(0, 3, 1, 2)

            fid = FrechetInceptionDistance(normalize=True)
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)
            score = fid.compute()

            scores.append(float(score))
            print(f"{bit}bit-{std}std FID score: {float(score)}")

        writer.writerow([bit] + scores)
