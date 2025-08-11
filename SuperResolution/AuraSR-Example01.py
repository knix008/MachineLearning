from aura_sr import AuraSR
import datetime

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

import requests
from io import BytesIO
from PIL import Image

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

image = load_image_from_url("https://mingukkang.github.io/GigaGAN/static/images/iguana_output.jpg").resize((256, 256))
upscaled_image = aura_sr.upscale_4x_overlapped(image)

upscaled_image.save(f"AuraSR-Example01_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")