import torch
from diffusers import Flux2Pipeline
from datetime import datetime

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype
)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"flux-klein_{timestamp}.png"
image.save(filename)
