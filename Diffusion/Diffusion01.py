from diffusers import DiffusionPipeline
import torch

model_id = "model/2880"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of a man wears cottonXcasualXhoodieXblue hoodie"
image = pipe(prompt, num_inference_steps=50, guidance_scale=9, width=512,height=512).images[0]

image.save("person-cottonXcasualXhoodieXblue-hoodie.png")