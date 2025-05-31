import torch
import os
from diffusers import AutoPipelineForText2Image

# Disable warning messages.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Checking GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

if device.type == "cuda":
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
    )
else:
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
    )

pipeline = pipeline.to(device)

# Generate an image
generator = [torch.Generator(device=device)]
image = pipeline(
    "The Taj Mahal on the Moon with Earth in sight.", generator=generator
).images[0]

image.show()

# Save the generated image
image.save("TajMahalOnMoon.png")
