import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from datetime import datetime
import os
import sys
import gc

# Define device type and data type
device_type = "cpu"
data_type = torch.float32

# Load input image using absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(script_dir, "default.png")
input_image = load_image(input_image_path)
print(f"Input image loaded: {input_image_path}")


print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=data_type
)
pipe.to(device_type)

print("Appliying optimizations...")
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
print("Model loaded... ")


def main():
    global pipe, input_image
    print("Generating image...")
    image = pipe(
        image=input_image,
        prompt="Add a beach background with palm trees and a bright sunny sky, vivid colors, high detail",
        width=768,
        height=1024,
        guidance_scale=2.5,
        num_inference_steps=20,
        generator=torch.Generator(device=device_type).manual_seed(42),
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
    del input_image
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Resources released!")
    return 0

if __name__ == "__main__":
    main()
