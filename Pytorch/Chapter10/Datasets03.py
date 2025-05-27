import torch
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Checking GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

from datasets import load_dataset, Image, Dataset

dataset = load_dataset("huggan/selfie2anime", split="train")
#print(dataset)

from torchvision import transforms

IMAGE_SIZE = 128
preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

#img = dataset["imageB"][0]
#print(img)

def transform(examples):
    images = [preprocess(image) for image in examples["imageB"]]
    return {"images": images}

dataset.set_transform(transform)
#print("> Transformed ", dataset)
