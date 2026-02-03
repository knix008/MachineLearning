import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from datetime import datetime
import os

print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("Model loaded!")

input_image = load_image("./default.jpg")

image = pipe(
    image=input_image,
    prompt="Add a beach background with palm trees and a bright sunny sky.",
    guidance_scale=2.5,
).images[0]

# Save with program name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"{script_name}_{timestamp}.png")
print(f"Image saved! : {script_name}_{timestamp}.png")

# Cleanup resources
del image
del pipe
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print("Resources released!")
