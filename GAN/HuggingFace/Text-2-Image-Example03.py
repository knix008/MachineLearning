import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import warnings
warnings.filterwarnings("ignore")

import torch
import time
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
prompt1 = "A dog is sitting on a bench in the yard"
prompt2 = "A Woman is walking in the street"

start = time.time()
image = pipe(prompt=prompt1).images[0]
print("Saving... : ", f"{prompt1}.png")
image.save(f"{prompt1}.png")
end = time.time()
print("Elapsed : ", end - start)

start = time.time()
image = pipe(prompt=prompt2).images[0]
print("Saving... : ", f"{prompt2}.png")
image.save(f"{prompt2}.png")
end = time.time()
print("Elapsed : ", end - start)