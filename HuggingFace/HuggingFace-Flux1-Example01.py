import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

# Disable all other warning message.
import warnings
warnings.filterwarnings("ignore")

import torch
from diffusers import FluxPipeline

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using... : ", device)

# Get the pretrained model.
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.to(device)

# Generate image.
prompt = "A cat holding a sign that says hello world"
print("The prompt : ", prompt)

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# Save image.
print("Saving the image...")
image.save("flux-dev.png")
print("Done!!!")

# NOTICE : You need to run "pip install sentencepiece" for tokenizer installation.