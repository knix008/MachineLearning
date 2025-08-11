import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import datetime

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load pipeline
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
print("모델을 CPU로 로딩 완료!")

# Load a control image
control_image = load_image("image.webp")

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

image.save(
    f"Flux1-ControlNet-Upscaler-Example01_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
)
