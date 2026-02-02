import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from datetime import datetime
import os

repo_id = "black-forest-labs/FLUX.2-dev"
device="cuda"
torch_dtype = torch.bfloat16

# Load model
pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
print("Model loaded!")

prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style cute korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. She is wearing a red bikini and her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look."

# Use the pipe directly - it handles text encoding internally
# Generator device should match where the model's first layer is
device_for_generator = device

image = pipe(
    prompt=prompt,
    generator=torch.Generator(device=device_for_generator).manual_seed(42),
    num_inference_steps=28,  # 28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

# Save with program name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"{script_name}_{timestamp}.png")