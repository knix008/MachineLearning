import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

# Disable warning messages.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"

text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to(device)

# generate an image
prompt = "high resolution, a photograph of an astronaut riding a horse"
image = text2img_pipe(prompt=prompt).images[0]

# show image
plt.imshow(image)
plt.show()
