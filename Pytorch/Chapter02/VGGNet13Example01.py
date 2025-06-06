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

ddir = "hymenoptera_data"

# Data normalization and augmentation transformations for train dataset
# Only normalization transformation for validation dataset
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
    k: torch.utils.data.DataLoader(
        img_data[k], batch_size=8, shuffle=True, num_workers=2
    )
    for k in ["train", "val"]
}
dset_sizes = {x: len(img_data[x]) for x in ["train", "val"]}
# Check if GPU is available and set device accordingly
dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import ast

with open("./imagenet1000_clsidx_to_labels.txt") as f:
    classes_data = f.read()
classes_dict = ast.literal_eval(classes_data)
print({k: classes_dict[k] for k in list(classes_dict)[:5]})


def imageshow(img, text=None):
    img = img.numpy().transpose((1, 2, 0))
    avg = np.array([0.490, 0.449, 0.411])
    stddev = np.array([0.231, 0.221, 0.230])
    img = stddev * img + avg
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if text is not None:
        plt.title(text)


def visualize_predictions(pretrained_model, max_num_imgs=4):
    torch.manual_seed(1)
    was_model_training = pretrained_model.training
    pretrained_model.eval()
    pretrained_model.to(dvc)
    imgs_counter = 0

    with torch.no_grad():
        for i, (imgs, tgts) in enumerate(dloaders["val"]):
            imgs = imgs.to(dvc)
            ops = pretrained_model(imgs)
            _, preds = torch.max(ops, 1)
            plt.figure()
            for j in range(imgs.size()[0]):
                imgs_counter += 1
                ax = plt.subplot(max_num_imgs // 2, 2, imgs_counter)
                ax.axis("off")
                ax.set_title(f"pred: {classes_dict[int(preds[j])]}")
                imageshow(imgs.cpu().data[j])
                if imgs_counter == max_num_imgs:
                    pretrained_model.train(mode=was_model_training)
                    plt.show()
                    return
        pretrained_model.train(mode=was_model_training)


model = models.vgg13(pretrained=True)
visualize_predictions(model)
