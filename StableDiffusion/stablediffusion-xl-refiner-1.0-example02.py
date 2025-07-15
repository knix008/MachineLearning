import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")

file = "Chloe_wine_DSC_1335-copy.jpg"
init_image = load_image(file).convert("RGB")
prompt = "blue bikini, ultra high definition photo realistic portrait, similar to a photo"
image = pipe(prompt, image=init_image).images
image[0].save("blue-bikini-woman.png")
