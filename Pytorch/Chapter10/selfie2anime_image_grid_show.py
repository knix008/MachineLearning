from diffusers.utils import make_image_grid
from datasets import load_dataset
from diffusers.utils import make_image_grid
import matplotlib.pyplot as plt

DATASET_PATH = "./data/selfie2anime/train/imageB"


def show_image_grid(dataset):
    if not dataset:
        raise ValueError("Dataset is empty or not loaded properly.")
    # Assuming dataset["image"] is a list or numpy array of images
    grid = make_image_grid(dataset["image"][:16], rows=4, cols=4)
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


def loading_dataset(dataset_path):
    dataset = load_dataset(DATASET_PATH, split="train")
    return dataset


def main():
    DATASET_PATH = "./data/selfie2anime/train/imageB"
    dataset = loading_dataset(DATASET_PATH)
    show_image_grid(dataset)


if __name__ == "__main__":
    print("> Showing image grid from the dataset...")
    main()
