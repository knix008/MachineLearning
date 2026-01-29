import torch
from diffusers import Flux2Pipeline
from datetime import datetime
import os

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

device = "cpu"
device_type = torch.float32

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=device_type
).to(device)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing()  # Slice attention computation
print("모델 로딩 완료!")

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)

prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style cute korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."


image = pipe(
    prompt=prompt,
    sigmas=TURBO_SIGMAS,
    guidance_scale=2.5,
    height=1024,
    width=1024,
    num_inference_steps=8,
).images[0]

# Generate filename with script name and current date and time
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{script_name}_{timestamp}.png"

image.save(output_filename)
print(f"이미지가 저장되었습니다: {output_filename}")
