import torch
import os
from datetime import datetime
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

# 2. Generate Image
height = 1024
width = 1024
num_inference_steps = 9
guidance_scale = 0.0
seed = 42

image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=num_inference_steps,  # This actually results in 8 DiT forwards
    guidance_scale=guidance_scale,  # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_name = os.path.splitext(os.path.basename(__file__))[0]
filename = f"{script_name}_{timestamp}_{width}x{height}_gs{guidance_scale}_step{num_inference_steps}_seed{seed}.png"

image.save(filename)
print(f"이미지 저장됨: {filename}")
