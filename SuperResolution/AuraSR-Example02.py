from aura_sr import AuraSR
from PIL import Image
import datetime

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

image = Image.open("image.webp").resize((512, 512))
upscaled_image = aura_sr.upscale_4x_overlapped(image)
upscaled_image.save(f"AuraSR-Example02_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")