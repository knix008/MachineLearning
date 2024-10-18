import torch
from diffusers import FluxPipeline

import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
access_token = "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"
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using... : ", device)

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token=access_token)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    #generator=torch.Generator("cpu").manual_seed(0)
    generator=torch.Generator(device).manual_seed(0)
).images[0]

file_name = "flux-dev.jpg"
image.save(file_name)
image.show()