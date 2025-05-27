import torch
from torchvision import transforms
import os
from datasets import load_dataset, Image

# Disable warning messages.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"
# Checking GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

dataset = load_dataset("huggan/selfie2anime", split="train")
dataset.save_to_disk("./selfie2anime")
#print(dataset['imageB'][0])

images = transforms.ToPILImage(dataset['imageB'][0])
print(images)
