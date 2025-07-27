import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import datetime

# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
pipe.enable_vae_slicing()  # Slice VAE computation
print("모델 로딩 완료!")

# Load a control image
control_image = load_image(
    "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/examples/input.jpg"
)

w, h = control_image.size

# Upscale x4
control_image = control_image.resize((w * 4, h * 4))

image = pipe(
    prompt="",
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28,
    guidance_scale=3.5,
    height=control_image.size[1],
    width=control_image.size[0],
).images[0]

image.save(f"Flux1-dev-ControlNet-Upscaler-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
