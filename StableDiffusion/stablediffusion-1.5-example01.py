import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import os
import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

# Disable warning messages.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"


def init_pipeline():
    if device == "cuda":
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(device)
    else:
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to(device)
    return text2img_pipe


def generate_image(pipe, prompt):
    start = time.time()
    image = pipe(prompt=prompt).images[0]
    end = time.time()
    result = datetime.timedelta(seconds=end - start)
    print(result)
    # show image
    plt.imshow(image)
    plt.show()
    return image


if __name__ == "__main__":
    # generate an image
    prompt = "high resolution, a photograph of an astronaut riding a horse"
    text2img_pipe = init_pipeline()
    image = generate_image(text2img_pipe, prompt)
    image.save("astronaut_riding_horse.png")
