import os
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
import datetime

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("default.jpg").convert("RGB")

prompt = "change the image to 8k, high quality, high detail, wearing dark blue bikini, masterpiece, realistic, good anatomy, with good fingers."

inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save(
        f"Qwen-Example03_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    print(
        "image saved at",
        os.path.abspath(
            f"Qwen-Example03_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ),
    )
