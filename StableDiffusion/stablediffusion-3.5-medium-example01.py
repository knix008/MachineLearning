import torch
from diffusers import StableDiffusion3Pipeline
import os

# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
access_token = ""

def run():
    #pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
    #pipe = pipe.to("cuda")

    pipe = StableDiffusion3Pipeline.from_single_file("./models/sd3.5_medium.safetensors", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(
        "A cat holding a sign that says hello world",
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]

    image.show()
    image.save("result.jpg")

if __name__ == "__main__":
    run()