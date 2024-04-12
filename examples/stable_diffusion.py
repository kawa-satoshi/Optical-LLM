import sys
import os
import copy
from pathlib import Path
from shutil import rmtree

from diffusers import DiffusionPipeline

current_dir = os.path.join(Path().resolve())
sys.path.append(str(current_dir) + "/../")

from models.converter import analog_convert

dir = Path("result")
if dir.exists():
    rmtree(dir)
dir.mkdir()

model_id = "CompVis/ldm-text2im-large-256"
prompt = "A painting of a squirrel eating a burger"

ldm = DiffusionPipeline.from_pretrained(model_id).to("cuda")
original_image = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images[0]
original_image.save(dir / "original.png")

targets = ["k_proj", "q_proj", "v_proj"]
bits = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
stds = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

for bit in bits:
    for std in stds:
        ldm_analog = copy.deepcopy(ldm)
        analog_convert(
            ldm_analog.bert,
            bit,
            bit,
            bit,
            std,
            targets,
            convert_linear=True,
            convert_conv2d=True,
            type="affine",
        )
        ldm_analog.to("cuda")

        img = ldm_analog([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6).images[0]
        img.save(dir / f"{bit}bit-{std}std.png")
