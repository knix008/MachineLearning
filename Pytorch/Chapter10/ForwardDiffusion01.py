from PIL import Image
import numpy as np

img = Image.open("./taj.png")
#img.show()
# 256 is the max px value, 3 is num_channels
noise = 256*np.random.rand(*img.size, 3)
noisy_img = ((img + noise)/2).astype(np.uint8)
img_noise = Image.fromarray(noisy_img)
#img_noise.show()
img_noise.save("./noisy_taj.png")