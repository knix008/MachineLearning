import torch
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.models import EfficientNet_B7_Weights
import urllib

sample = "dog.jpg"


def get_model():
    """
    Load the EfficientNet_B7 model with pretrained weights.
    """
    # Load the Inception v3 model with pretrained weights
    model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
    print("> Initialized EfficientNet_B7 model with default weights.")
    return model.eval()  # Set the model to evaluation mode


def download_sample_image():
    """
    Download a sample image from the pytorch website.
    """
    # Download an example image from the pytorch website
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", sample)
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)


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
    print("> EfficientNet_B7 Example")
    # Get the model with default weights
    model = get_model()  # Load the model
    download_sample_image()  # Download the sample image
    input_batch = transform_image(sample)  # Transform the image
    categories = read_categories()
    device = get_device_name()  # Get the device
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
