import torch
from modelscope import FluxPipeline
import datetime

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)

pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU.
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU sequentially.
pipe.enable_attention_slicing() #save some VRAM by slicing the attention layers.
pipe.enable_flash_attention_2() #save some VRAM by using flash attention 2.
pipe.enable_xformers_memory_efficient_attention() #save some VRAM by using xformers memory efficient attention.
pipe.enable_torch_compile() #enable torch compile for faster inference.

prompt = "A frog holding a sign that says hello world"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]

image.save(f"Flux1_Krea_Dev-Example01-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")