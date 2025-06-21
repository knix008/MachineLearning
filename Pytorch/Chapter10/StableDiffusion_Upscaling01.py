import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import time
import datetime

def main():
    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")
    IMAGE_PATH = "sample_face.jpg"

    low_res_img = Image.open(IMAGE_PATH).convert("RGB")
    low_res_img = low_res_img.resize((512, 512))
    low_res_img.save("low_res_cat.png")

    prompt = "A high-resolution image of a baautiful face, ultra detailed and vibrant"
    start = time.time()
    upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
    end = time.time()
    print(f"Upscaling took {datetime.timedelta(seconds=end - start)}")
    upscaled_image.save("upscaled_image.png")

if __name__ == "__main__":
    print("Upscaling example...")
    main()