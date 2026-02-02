import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from datetime import datetime
import os

device = "cpu"

repo_id = "black-forest-labs/FLUX.2-dev"  # Standard model
torch_dtype = torch.float32  # Use float32 for CPU

# Load model on CPU with low memory usage
pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)

# Enable sequential CPU offload - moves layers to GPU only when needed
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
print("모델 로딩 완료!")

prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a half-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

# Use the pipe directly - it handles text encoding internally
# Generator device should match where the model's first layer is
device_for_generator = "cpu"
seed = 42
guidance_scale = 4

image = pipe(
    prompt=prompt,
    generator=torch.Generator(device=device_for_generator).manual_seed(seed),
    num_inference_steps=28,  # 28 steps can be a good trade-off
    guidance_scale=guidance_scale,
).images[0]

# Save with program name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"{script_name}_{timestamp}_seed{seed}_cfg{guidance_scale}.png")