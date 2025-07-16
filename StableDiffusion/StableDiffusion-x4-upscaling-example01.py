import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# load low resolution image
image = Image.open("SuperResolution-XL-Turbo-Result02.jpg").convert("RGB")
prompt = "Ultra high resolution, hightly detailed, photrealstic, 8k resolution, high quality, masterpiece, cinematic, award winning, trending on artstation, unreal engine, octane render, hyper realistic, intricate details, sharp focus, depth of field, volumetric lighting, realistic shadows, high dynamic range, ultra detailed textures"

# resize image to 512x512 and upscale
image = image.resize((512, 512), Image.LANCZOS)
upscaled_image = pipeline(prompt=prompt, image=image).images[0]
upscaled_image.save("SuperResolution-XL-Turbo-Result03.png")
