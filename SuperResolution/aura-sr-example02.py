from aura_sr import AuraSR
import time
import requests
from io import BytesIO
from PIL import Image

aura_sr = AuraSR.from_pretrained()

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

def upscale():
    #image = load_image_from_url("https://mingukkang.github.io/GigaGAN/static/images/iguana_output.jpg").resize((256, 256))
    #image.save("resized.jpg")
    #image = Image.open("Test01.jpg").resize((256, 256))
    image = Image.open("Test01.jpg")
    start = time.time()
    upscaled_image = aura_sr.upscale_4x(image)
    end = time.time()
    print(f"{end - start:.5f} sec")
    upscaled_image.save("upscaled.jpg")

if __name__ == "__main__":
    upscale()