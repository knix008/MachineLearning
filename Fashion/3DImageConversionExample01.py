from PIL import Image
import numpy as np
import requests
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas", low_cpu_mem_usage=True
)

url = "Cats_Test01.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(url)
image = image.convert("RGB")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
#depth.show()
print(">Depth estimation completed.")
depth.save("Cats_Test01_Depth.jpg")
