import cv2
import numpy as np
from PIL import Image
import gradio as gr
import random

# NOTE: As of June 2024, YOLO 11 does not exist. This uses YOLOv11 (Ultralytics) as the latest available.
from ultralytics import YOLO

# Load the YOLOv11 model (replace with your own model if available)
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model.to(
    "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
)  # Use GPU if available

def get_unique_color(idx):
    random.seed(idx)  # Deterministic color for each detection in one run
    return tuple(random.randint(0, 255) for _ in range(3))

def detect_objects(pil_img):
    # Convert PIL image to OpenCV format
    image = np.array(pil_img)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    results = model(image)
    output_img = image.copy()
    labels = []
    index = 0
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0
        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
        label = (
            results[0].names[cls_id]
            if hasattr(results[0], "names") and cls_id != -1
            else str(cls_id)
        )
        labels.append(f"{label} ({conf:.2f})")
        color = get_unique_color(index)
        # Draw bounding box and label
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output_img,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        index += 1
    pil_output = Image.fromarray(output_img)
    detected = "\n".join(labels) if labels else "No objects detected."
    return pil_output, detected


demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Detected Objects"),
        gr.Textbox(label="Labels"),
    ],
    title="YOLO Object Detection (YOLOv11)",
    description="Upload an image. The app detects objects with YOLO (latest version available) and displays results with bounding boxes and labels.",
)

if __name__ == "__main__":
    demo.launch()
