from aura_sr import AuraSR
from PIL import Image
import datetime

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

image = Image.open("default.png").resize((256, 256))
upscaled_image = aura_sr.upscale_4x_overlapped(image)
upscaled_image.save(f"AuraSR-Example03_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")