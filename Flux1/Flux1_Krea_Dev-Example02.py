import torch
from diffusers import FluxPipeline
import datetime

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)

pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU.
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU sequentially.
pipe.enable_attention_slicing() #save some VRAM by slicing the attention layers.


prompt = "A frog holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]

image.save(f"flux-krea-dev-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")