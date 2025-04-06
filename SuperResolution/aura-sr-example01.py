from aura_sr import AuraSR

aura_sr = AuraSR.from_pretrained()

import requests
from io import BytesIO
from PIL import Image

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

image = load_image_from_url("https://mingukkang.github.io/GigaGAN/static/images/iguana_output.jpg").resize((256, 256))
image.save("resized.jpg")
upscaled_image = aura_sr.upscale_4x(image)
upscaled_image.save("upscaled.jpg")