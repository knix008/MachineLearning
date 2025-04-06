from aura_sr import AuraSR
import time
import requests
from io import BytesIO
from PIL import Image

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

def upscale():
    image = load_image_from_url("https://mingukkang.github.io/GigaGAN/static/images/iguana_output.jpg").resize((256, 256))
    image.save("resized2.jpg")
    start = time.time()
    upscaled_image = aura_sr.upscale_4x_overlapped(image)
    end = time.time()
    print(f"{end - start:.5f} sec")
    upscaled_image.save("upscaled2.jpg")
    
if __name__ == "__main__":
    upscale()