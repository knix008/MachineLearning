import torch
from diffusers import FluxPipeline
import datetime
import os
import warnings

# Disable all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Disable Hugging Face symlinks warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)

pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU.
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU sequentially.
pipe.enable_attention_slicing() #save some VRAM by slicing the attention layers.


prompt = "A woman wearing a red bikini, walking on a beach, looking at viewer, high quality, realistic, high detail, 8k, cinematic lighting"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]

image.save(f"flux1-krea-dev-example02-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")