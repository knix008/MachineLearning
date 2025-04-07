from diffusers import StableDiffusionPipeline
import torch

model_id = "prompthero/openjourney"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
#prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"
#prompt = "retro serie of different cars with different colors and shapes"
#prompt = "woman walking on a beach, swimsuit, daylight, oriental, long hair, sunset, 5 fingers"
prompt = "woman fashion design pencil sketch, modern, swimsuit, long legs, sunglasses"
image = pipe(prompt).images[0]
#image.save("./retro_cars.png")
image.save("./design01.png")