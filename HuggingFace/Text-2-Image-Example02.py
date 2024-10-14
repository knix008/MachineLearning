import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import warnings
warnings.filterwarnings("ignore")

import torch
from diffusers import DiffusionPipeline

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using... : ", device)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True, 
                                         variant="fp16")
pipe.to(device)

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
prompt = "a dog is sitting on a bench in the yard"
image = pipe(prompt=prompt).images[0]
print("Saving... : ", f"{prompt}.png")
image.save(f"{prompt}.png")