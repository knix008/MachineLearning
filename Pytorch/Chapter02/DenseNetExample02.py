import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.models import (
    DenseNet169_Weights,
    DenseNet161_Weights,
    DenseNet121_Weights,
    DenseNet201_Weights,
)
import urllib


def get_model():
    """
    Load the DenseNet model with pretrained weights.
    """

    # Load the DenseNet model with pretrained weights
    # model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    # model = models.densenet161(weights=DenseNet161_Weights.DEFAULT)
    # model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
    model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
    print("> Initialized DenseNet201 model with default weights.")
    return model.eval()  # Set the model to evaluation mode


def transform_image(image_path):
    """
    Load and transform the image for the model.
    """
    input_image = Image.open(image_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    return input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model


# move the input and model to GPU for speed if available
def get_device_name():
    """
    Get the name of the device being used.
    """
    if torch.cuda.is_available():
        print("> The GPU device name : ", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        return "cpu"


# Run the model on the input image
def run_model(model, input_batch):
    """
    Run the model on the input batch.
    """
    # Set the model to evaluation
    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)
    return probabilities  # Return the probabilities for further processing


def read_categories():
    """
    Read the categories from the ImageNet classes file.
    """
    # Read the categories
    with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories


def show_top_categories(categories, probabilities):
    """
    Show the top categories and their probabilities.
    """
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


def main():
    """
    Main function to run the GoogleNet example.
    """
    print("> DenseNet Example")
    sample = "cat.jpg"
    # Get the model with default weights
    model = get_model()  # Load the model
    input_batch = transform_image(sample)  # Transform the image
    categories = read_categories()
    device = get_device_name()  # Get the device
    # device = "cpu"
    print("> Using device:", device)
    if device == "cpu":
        print("> Using CPU for inference.")
    else:
        print("> Using GPU for inference.")
        model.to(device)  # Move the model to GPU
        input_batch = input_batch.to(device)  # Move the input to GPU

    probabilities = run_model(model, input_batch)  # Run the model on the input batch
    show_top_categories(categories, probabilities)  # Show the top categories
    print("> Inference completed.")


if __name__ == "__main__":
    main()  # Run the main function
