import numpy as np
from PIL import Image

image_file = 'sample.jpg'

img = Image.open(image_file)
try:
    img.show()
except:
    print("Cannot diplay image!!!")