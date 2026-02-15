import os
from datetime import datetime
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "shauray/FLUX-UNCENSORED-merged", torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

prompt = "A mystic cat with a sign that says hello world!"
guidance_scale = 3.5
num_inference_steps = 28
seed = 0

image = pipe(
    prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=torch.manual_seed(seed),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_name = os.path.splitext(os.path.basename(__file__))[0]
filename = f"{script_name}_{timestamp}_gs{guidance_scale}_step{num_inference_steps}_seed{seed}.png"

image.save(filename)
print(f"이미지 저장됨: {filename}")
