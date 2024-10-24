import torch
from diffusers import DiffusionPipeline
import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
import warnings
warnings.filterwarnings("ignore")

def main(prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using... : ", device)

    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipeline.to(device)
    img = pipeline(prompt).images[0]
    img.save(f"{prompt}.jpg")
    
if __name__ == "__main__":
    main("An image of a squirrel in Picasso style")