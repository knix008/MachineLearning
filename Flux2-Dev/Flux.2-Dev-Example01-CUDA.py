import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" #quantized text-encoder and DiT. VAE still in bf16
device = "cuda:0"
torch_dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. She is wearing a red bikini and her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look."

image = pipe(
    prompt = prompt,
    #image=[cat_image] #optional multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")