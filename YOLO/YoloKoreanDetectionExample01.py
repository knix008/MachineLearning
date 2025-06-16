import cv2
import numpy as np
import pytesseract
from PIL import Image
import gradio as gr

# NOTE: There is no YOLO 11 as of 2024. We use latest YOLOv8 (Ultralytics) as a stand-in.
from ultralytics import YOLO

# Load YOLOv8 model (for text detection, use a model trained for text/ocr tasks)
model = YOLO('yolov8n.pt')  # Replace with a text-detection YOLO model for better results

def detect_and_recognize_text(pil_img):
    image = np.array(pil_img)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    results = model(image)
    recognized_texts = []
    image_out = image.copy()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop).strip()
        if text:
            recognized_texts.append(text)
            cv2.rectangle(image_out, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(image_out, text, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    display_image = Image.fromarray(image_out)
    all_text = "\n".join(recognized_texts) if recognized_texts else "No text found."
    return display_image, all_text

demo = gr.Interface(
    fn=detect_and_recognize_text,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Image with Text Boxes"),
        gr.Textbox(label="Recognized Text")
    ],
    title="Text Detection (YOLO + Tesseract)",
    description="Upload an image. The app detects text using YOLO (latest version) and recognizes it with Tesseract OCR."
)

if __name__ == "__main__":
    demo.launch()