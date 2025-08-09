import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import datetime

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
url = "default.jpg"
init_image = load_image(url).convert("RGB")
prompt = "8k, high detail, high quality, photo realistic, masterpiece, best quality, dark blue bikini"
image = pipe(prompt, image=init_image).images[0]
image.save(f"SDXL-1.0-Refiner-Example01-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")