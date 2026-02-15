import torch
import os
from diffusers import FluxPipeline
from datetime import datetime

pipe = FluxPipeline.from_pretrained(
    "kpsss34/FHDR_Uncensored", torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

prompt = "a women..."
height = 1024
width = 1024
guidance_scale = 4.0
num_inference_steps = 40
max_sequence_length = 512
seed = 0

image = pipe(
    prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    max_sequence_length=max_sequence_length,
    generator=torch.Generator("cpu").manual_seed(seed),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_name = os.path.splitext(os.path.basename(__file__))[0]
filename = f"{script_name}_{timestamp}_{width}x{height}_gs{guidance_scale}_step{num_inference_steps}_seed{seed}_msl{max_sequence_length}.png"

image.save(filename)
print(f"Image saved: {filename}")
