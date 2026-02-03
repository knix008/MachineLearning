import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from datetime import datetime
import os

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("Model loaded!")

# Load input image using absolute path
input_image_path = os.path.join(script_dir, "default.png")
input_image = load_image(input_image_path)
print(f"Input image loaded: {input_image_path}")

image = pipe(
    image=input_image,
    prompt="Add a beach background with palm trees and a bright sunny sky.",
    guidance_scale=2.5,
).images[0]

# Save with program name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(script_dir, f"{script_name}_{timestamp}.png")
image.save(output_path)
print(f"Image saved! : {output_path}")

# Cleanup resources
del image
del pipe
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print("Resources released!")
