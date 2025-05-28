import torch
import torchvision
import matplotlib.pyplot as plt
import kagglehub

# Download latest version
path = kagglehub.dataset_download("nadinpethiyagoda/vehicle-dataset-for-yolo", force_download=False)
print("Path to dataset files:", path)

def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])

#data = torchvision.datasets.StanfordCars(root='.', download=True)
#show_images(data)