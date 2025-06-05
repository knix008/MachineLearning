import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

torch.use_deterministic_algorithms(True)

# Creating a local data directory
ddir = "hymenoptera_data"
# Data normalization and augmentation transformations
# for train dataset
# Only normalization transformation for validation dataset
# The mean and std for normalization are calculated as the
# mean of all pixel values for all images in the training
# set per each image channel - R, G and B
data_transformers = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.490, 0.449, 0.411], [0.231, 0.221, 0.230]),
        ]
    ),
}

img_data = {
    k: datasets.ImageFolder(os.path.join(ddir, k), data_transformers[k])
    for k in ["train", "val"]
}

dloaders = {
    k: torch.utils.data.DataLoader(img_data[k], batch_size=8, shuffle=True)
    for k in ["train", "val"]
}
dset_sizes = {x: len(img_data[x]) for x in ["train", "val"]}
classes = img_data["train"].classes

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("> Loading the dataset and check the device : ", dvc)


def imageshow(img, text=None):
    img = img.numpy().transpose((1, 2, 0))
    avg = np.array([0.490, 0.449, 0.411])
    stddev = np.array([0.231, 0.221, 0.230])
    img = stddev * img + avg
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if text is not None:
        plt.title(text)
    plt.show()

print("> Showing a sample image from the dataset")
# Generate one train dataset batch
imgs, cls = next(iter(dloaders["train"]))
# Generate a grid from batch
grid = torchvision.utils.make_grid(imgs)
imageshow(grid, text=[classes[c] for c in cls])
