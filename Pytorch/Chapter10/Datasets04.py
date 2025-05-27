import torch
import os
from PIL import Image
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Checking GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Device : ", device)

from datasets import load_dataset

dataset = load_dataset("huggan/selfie2anime", split="train")
#print(dataset)

from torchvision import transforms

IMAGE_SIZE = 128
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.5], [0.5]),
        transforms.ToTensor(),
    ]
)

#img = dataset["imageB"][0]
#print(img)

def transform(examples):
    images = [preprocess(image) for image in examples["imageB"]]
    return {"images": images}

dataset.set_transform(transform)
print("> Transformed ", dataset)

BSIZE = 16 # batch size
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BSIZE, shuffle=True)

from diffusers import DDPMScheduler
noise_scheduleer = DDPMScheduler(num_train_timesteps=1000)

clean_images = next(iter(train_dataloader))["images"]
# Sample noise to add to the images
noise = torch.randn(clean_images.shape, device=clean_images.device)
bs = clean_images.shape[0]

# Sample a random timestep for each image
timesteps = torch.range(10, 161, 10, dtype=torch.int)

# Add noise to the clean images according to the noise magnitude at each timestep
# (this is the forward diffusion process)
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
