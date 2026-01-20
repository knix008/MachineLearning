import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=dtype)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]

# Generate filename with program name, date and time
program_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{program_name}_{timestamp}.png"
image.save(output_filename)