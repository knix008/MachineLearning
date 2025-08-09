import PIL
import requests
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
import datetime

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "turn the photo into 8k, high detail, high quality, photo realistic, masterpiece, best quality, wearing dark blue bikini photo."
input_image = PIL.Image.open("default.jpg")
image = pipe(
    prompt, image=input_image, num_inference_steps=50, image_guidance_scale=1
).images[0]

image.save(
    f"StableDiffusion-Instruct-Pix2Pix-Example01-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
)
