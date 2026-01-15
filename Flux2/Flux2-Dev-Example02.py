import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from datetime import datetime

repo_id = "black-forest-labs/FLUX.2-dev"  # Standard model
torch_dtype = torch.float16  # Use float16 for GPU efficiency

# Load model with CPU offloading for memory management
pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)

# Enable sequential CPU offload - moves each layer to GPU only when needed
# This is the most aggressive offloading, minimizing VRAM usage
pipe.enable_sequential_cpu_offload()

# Enable attention slicing to reduce memory during attention computation
pipe.enable_attention_slicing(slice_size="auto")

# Enable VAE slicing for lower memory VAE decoding
pipe.enable_vae_slicing()

# Enable VAE tiling for very large images (reduces memory for VAE)
pipe.enable_vae_tiling()


print("모델 로딩 완료!")
print("CPU offloading enabled with the following optimizations:")
print("  - Sequential CPU offload: layers move to GPU only when needed")
print("  - Attention slicing: reduces memory during attention computation")
print("  - VAE slicing: processes VAE in slices to reduce peak memory")
print("  - VAE tiling: enables tiled VAE decoding for lower memory usage")


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
