from aura_sr import AuraSR
import time
import requests
from io import BytesIO
from PIL import Image
import os

# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

def upscale():
    image = Image.open("Test04.jpg")
    start = time.time()
    upscaled_image = aura_sr.upscale_4x_overlapped(image)
    end = time.time()
    print(f"{end - start:.5f} sec")
    upscaled_image.save("4x.jpg")

    image = Image.open("4x.jpg")
    start = time.time()
    upscaled_image = aura_sr.upscale_4x_overlapped(image)
    end = time.time()
    print(f"{end - start:.5f} sec")
    upscaled_image.save("16x.jpg")
    
if __name__ == "__main__":
    upscale()