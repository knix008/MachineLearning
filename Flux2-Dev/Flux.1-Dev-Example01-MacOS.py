import torch
from diffusers import FluxPipeline
from datetime import datetime
import os

# ============== Parameters ==============
prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style cute and lovely korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

width = 768
height = 1024
guidance_scale = 4.0
num_inference_steps = 28
seed = 400
# ========================================
# Set device and data type
device = "mps"
dtype = torch.bfloat16

print("Loading model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
).to(device)

# Enable memory optimizations
#pipe.enable_model_cpu_offload()
#pipe.enable_attention_slicing()
#pipe.enable_sequential_cpu_offload() # More useful for very large models, more memory for GPU, less memory for CPU, Slower inference.
print("Model loaded!")

print("\nGenerating image with:")
print(f"  Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt: {prompt}")
print(f"  Size: {width}x{height}")
print(f"  Steps: {num_inference_steps}")
print(f"  Guidance: {guidance_scale}")
print(f"  Seed: {seed}")
print()

# Generate image
image = pipe(
    prompt=prompt,
    width=width,
    height=height,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator(device=device).manual_seed(seed),
).images[0]

# Save with program name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{script_name}_{timestamp}_step{num_inference_steps}_seed{seed}.png"
image.save(filename)

print(f"Image saved: {filename}")
