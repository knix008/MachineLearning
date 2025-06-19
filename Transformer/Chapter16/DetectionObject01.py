from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

url = "http://images.cocodataset.org/val2017/000000439715.jpg"
image = Image.open(requests.get(url, stream=True).raw)


inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)



target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
 outputs, target_sizes=target_sizes, threshold=0.9)[0]


for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} \
 with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
