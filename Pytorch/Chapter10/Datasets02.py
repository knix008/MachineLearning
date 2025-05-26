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