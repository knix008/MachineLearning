import torch
import torchvision
import kagglehub
import os
from PIL import Image
import matplotlib.pyplot as plt

class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path, transform=None):
        self.images = [os.path.join(root_path, file) for file in os.listdir(root_path)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_file = self.images[index]
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image[None]


def show_images(dataset, num_samples=20, cols=4):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0].permute(1, 2, 0))  # Convert to HWC format for plotting
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # ​Download latest version
    image_path = r"vehicle dataset\train\images"
    
    path = kagglehub.dataset_download(
        "nadinpethiyagoda/vehicle-dataset-for-yolo", force_download=False
    )
    print("Path to dataset files:", path)
    train_image_path = os.path.join(path, image_path)
    #print("Training images path : ", train_image_path)
    stanfordcars = StanfordCars(
        train_image_path, transform=torchvision.transforms.ToTensor()
    )
    show_images(stanfordcars)
