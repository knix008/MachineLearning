import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_file_path = 'sample.jpg'

img = Image.open(image_file_path)
try:
    img.show()
except:
    print("Cannot diplay image!!!")