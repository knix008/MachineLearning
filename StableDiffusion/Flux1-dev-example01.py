import torch
from diffusers import FluxPipeline

# Load model with memory optimizations
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
pipe.enable_vae_slicing()  # Slice VAE computation

# Don't manually move to CPU - let sequential_cpu_offload handle device management

#prompt = "a futuristic city skyline at sunset, cyberpunk style"
prompt = "a futuristic city skyline, photorealistic, 8k resolution, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality"

# Reduce image size to save memory
image = pipe(
    prompt,
    height=1024,  # Reduced from 1024 to save memory
    width=1024,   # Reduced from 1024 to save memory
    guidance_scale=3.5,
    num_inference_steps=50,  # Reduced steps for faster generation
    max_sequence_length=256,  # Reduced sequence length
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev03.png")
