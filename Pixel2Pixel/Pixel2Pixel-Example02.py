import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import datetime

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

image = PIL.Image.open("default.jpg")

prompt = "8k, high quality, high detail, best quality, photo realistic"
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images

images[0].save(f"Pixel2Pixel-Example02_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")