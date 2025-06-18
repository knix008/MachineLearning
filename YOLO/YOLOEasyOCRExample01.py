import torch
import gradio as gr
import cv2
import numpy as np
import easyocr
from PIL import Image

# 1. YOLO 모델 로드 (일반 yolov5s 예시, 텍스트 박스 검출을 위한 커스텀 모델 추천)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# 2. EasyOCR 한글+영어
reader = easyocr.Reader(['ko', 'en'])

def detect_and_ocr(image):
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # YOLO로 객체(텍스트 영역) 검출
    results = model(img_rgb)
    boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    texts = []
    img_draw = img.copy()

    for *xyxy, conf, cls in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        crop = img_rgb[y1:y2, x1:x2]
        ocr_results = reader.readtext(crop)
        # 박스 그리기
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0,255,0), 2)
        # OCR 결과 텍스트 저장
        for (_, text, prob) in ocr_results:
            texts.append(f"{text} (conf: {prob:.2f})")

    # 결과 이미지 PIL 변환
    img_boxed = Image.fromarray(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    result_text = "\n".join(texts) if texts else "No text detected."

    return img_boxed, result_text

with gr.Blocks() as demo:
    gr.Markdown("## YOLO + OCR (한글/영문) 이미지 텍스트 인식")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="이미지 업로드", type="numpy")
            btn = gr.Button("텍스트 인식")
        with gr.Column():
            out_img = gr.Image(label="박스 처리된 이미지")
            out_text = gr.Textbox(label="인식된 텍스트", lines=15)
    btn.click(fn=detect_and_ocr, inputs=inp, outputs=[out_img, out_text])

if __name__ == "__main__":
    demo.launch()