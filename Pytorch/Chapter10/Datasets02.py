import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Checking GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

from datasets import load_dataset

dataset = load_dataset("huggan/selfie2anime", split="train")
print("> Loading datasets : ", dataset)

img = dataset["imageB"][0]

file = os.open("imageB-0.png", os.O_WRONLY | os.O_CREAT | os.O_BINARY)
os.write(file, img['bytes'])

import matplotlib.pyplot as plt
import cv2

print("> Showing image...")
img = cv2.imread('imageB-0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#show iage
plt.imshow(img)
plt.show()