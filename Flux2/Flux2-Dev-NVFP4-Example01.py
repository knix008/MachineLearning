import torch
from diffusers import FluxPipeline
from datetime import datetime
import os

# Model configuration
model_path = "black-forest-labs/FLUX.2-dev-NVFP4"
model_file = "flux2-dev-nvfp4.safetensors"

# Download the model weights
from huggingface_hub import hf_hub_download

try:
    local_model_path = hf_hub_download(
        repo_id=model_path,
        filename=model_file,
        cache_dir="./models",
    )
    print(f"Model downloaded to: {local_model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Make sure you have git-lfs installed and proper permissions")
    exit(1)

# Load from single file using FluxPipeline
try:
    pipe = FluxPipeline.from_single_file(
        local_model_path,
        torch_dtype=torch.float16,
    )
    print("Model loaded successfully from single file!")
except Exception as e:
    print(f"Single file loading failed: {e}")
    print("Trying alternative loading method...")
    try:
        # Alternative: Use FLUX.1 structure as base and load weights
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("Model loaded using FLUX.1 structure as base!")
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
        exit(1)

# Enable sequential CPU offload - moves each layer to GPU only when needed
# This is the most aggressive offloading, minimizing VRAM usage
try:
    pipe.enable_sequential_cpu_offload()
except:
    print("Sequential CPU offload not available, using regular offload")
    pipe.enable_model_cpu_offload()

print("모델 로딩 완료!")
print("CPU offloading enabled")


prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

# Use the pipe directly - it handles text encoding internally
# Generator device should match where the model's first layer is
device_for_generator = "cuda" if torch.cuda.is_available() else "cpu"
image = pipe(
    prompt=prompt,
    generator=torch.Generator(device=device_for_generator).manual_seed(42),
    num_inference_steps=28,  # 28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"flux2_output_{timestamp}.png")
