import torch
from diffusers import DiffusionPipeline
import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using... : ", device)

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to(device)
prompt = "An image of a squirrel in Picasso style"
img = pipeline(prompt).images[0]
img.save(f"{prompt}.jpg")