import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")
IMAGE_PATH = "generated_image.png"

low_res_img = Image.open(IMAGE_PATH).convert("RGB")
low_res_img = low_res_img.resize((512, 512))
low_res_img.save("low_res_cat.png")

prompt = "a white cat"

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("upscaled_image.png")
