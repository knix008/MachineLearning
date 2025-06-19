import torch
from PIL import Image
import requests
from transformers import ViTForImageClassification, ViTImageProcessor

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(requests.get(url, stream=True).raw)
    else:
        raise Exception("Failed to download image from URL: {}".format(url))

def model_and_processor():
    model_name = 'google/vit-base-patch16-224'
    # Load the model and processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # Load the pre-trained ViT model and image processor
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    processor = ViTImageProcessor.from_pretrained(model_name)
    return model, processor


def classify_image(image):
    model, processor = model_and_processor()
    model.eval()  # Set the model to evaluation mode
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Ensure the image is in RGB format
    image = image.resize((224, 224))  # Resize the image to the expected input size
    input = processor(images=image, return_tensors="pt")
    outputs = model(input.pixel_values)
    prediction = outputs.logits.argmax(-1)
    print("Class label: ", model.config.id2label[prediction.item()])

def main():
    url = 'http://images.cocodataset.org/val2017/000000439715.jpg'
    image = download_image(url)
    image.save("downloaded_image.jpg")  # Save the downloaded image
    classify_image(image)


if __name__ == "__main__":
    main()