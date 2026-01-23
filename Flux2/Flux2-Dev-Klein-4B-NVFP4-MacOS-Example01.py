import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os

device = "mps"
dtype = torch.float16  # Use float16 instead of bfloat16 for MPS compatibility

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipe = pipe.to(device)
#pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A highly realistic, 4k, high-quality vivid photo of a beautiful skinny animation-style girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, wearing a red bikini. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, avoiding an overly smooth or filtered look, to maintain a lifelike. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    #guidance_scale=4.0,
    num_inference_steps=10,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]

# Save with filename and datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_name = os.path.splitext(os.path.basename(__file__))[0]
output_path = f"{base_name}_{timestamp}.png"
image.save(output_path)
print(f"Image saved as: {output_path}")