import torch
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
import torchvision.transforms as T
import warnings
from PIL import Image
import time
import datetime

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_generater_id = "stabilityai/stable-diffusion-2-1-base"
model_upscaler_id = "stabilityai/stable-diffusion-x4-upscaler"


def initialize_generator_pipeline():
    # Load base Stable Diffusion 2.1 pipeline
    generator_pipe = StableDiffusionPipeline.from_pretrained(
        model_generater_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    generator_pipe.enable_model_cpu_offload()
    return generator_pipe


def generate_image(generator_pipe, prompt, seed, image_size=512):
    # Generate intermediate high-res image (PIL)
    low_res_image = generator_pipe(
        prompt=prompt,
        height=image_size,
        width=image_size,
        generator=seed,
        num_inference_steps=30,
    ).images[0]
    return low_res_image


def initialize_upscaler_pipeline():
    # Load the Stable Diffusion Upscale pipeline
    upscaler_pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_upscaler_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    upscaler_pipe.enable_model_cpu_offload()
    return upscaler_pipe


def upscale_image(upscaler_pipe, prompt, low_res_image):
    # Upscale the image
    upscaled_image = upscaler_pipe(prompt=prompt, image=low_res_image).images[0]
    return upscaled_image


IMAGE_SIZE = 512


def main():
    generator_pipe = initialize_generator_pipeline()
    prompt = "a vivid blue and yellow macaw in a tropical jungle, cinematic lighting, ultra-high detail"
    seed = torch.manual_seed(71)
    image = generate_image(generator_pipe, prompt, seed, image_size=IMAGE_SIZE)
    image.save("generated_image.png")
    del generator_pipe # Free up memory --> important for large models. Sometimes it will make OOM error.
    
    # Upscaling generated image
    print("> Upscaling generated image...")
    image = Image.open("generated_image.png").convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    upscaler_pipe = initialize_upscaler_pipeline()
    
    start = time.time()
    image = upscale_image(upscaler_pipe, prompt, image)
    end = time.time()
    print(f"> Upscaling completed in {datetime.timedelta(seconds=end - start)}")
    image.save("upscaled_image.png")
    del upscaler_pipe  # Free up memory


if __name__ == "__main__":
    print("> Generating and upscaling image...")
    main()
