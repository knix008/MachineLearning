import cv2
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download


# Download model
model_path = hf_hub_download(repo_id="Daniil-Domino/yolo11x-text-detection", filename="model.pt")

# Load model
model = YOLO(model_path)

# Inference
image_path = "image/test_image01.png"
image = cv2.imread(image_path).copy()
output = model.predict(image, conf=0.3)

# Draw bounding boxes
out_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
for data in output[0].boxes.data.tolist():
    xmin, ymin, xmax, ymax, _, _ = map(int, data)
    cv2.rectangle(out_image, (xmin, ymin), (xmax, ymax), color=(0, 0, 255), thickness=3)

# Display result
plt.figure(figsize=(15, 10))
plt.imshow(out_image)
plt.axis('off')
plt.show()