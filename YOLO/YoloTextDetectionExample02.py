import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import numpy as np
import gradio as gr
import pytesseract
import time

# Download and load model once
model_path = hf_hub_download(repo_id="Daniil-Domino/yolo11x-text-detection", filename="model.pt")
model = YOLO(model_path)
OCR_LANG = "kor+eng"

def detect_text(image):
    start_time = time.time()
    # Input image: PIL.Image, but YOLO/Opencv wants numpy
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    output = model.predict(image_bgr, conf=0.3)
    out_image = image_np.copy()
    ocr_results = []
    for box in output[0].boxes.data.tolist():
        xmin, ymin, xmax, ymax = map(int, box[:4])
        cv2.rectangle(out_image, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=3)
        crop = image_np[ymin:ymax, xmin:xmax]
        text = pytesseract.image_to_string(crop, lang=OCR_LANG).strip()
        if text:
            ocr_results.append(text)
    elapsed = time.time() - start_time
    # Display detected texts
    ocr_text = "\n".join(ocr_results) if ocr_results else "텍스트를 인식하지 못했습니다."
    return out_image, f"걸린 시간: {elapsed:.2f}초\n\n인식 결과:\n{ocr_text}"

interface = gr.Interface(
    fn=detect_text,
    inputs=gr.Image(type="pil", label="입력 이미지"),
    outputs=[
        gr.Image(type="numpy", label="텍스트 박스 표시 결과"),
        gr.Textbox(label="OCR 및 처리 시간"),
    ],
    title="YOLO + OCR (한글/영문) 텍스트 인식",
    description="입력 이미지를 올려주세요."
)

if __name__ == "__main__":
    interface.launch()