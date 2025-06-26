import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont # Pillow 라이브러리 임포트

try:
    FONT_PATH = 'NanumGothic-Regular.ttf'
    FONT = ImageFont.truetype(FONT_PATH, 30)
    print(f"'{FONT_PATH}' 폰트를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print(f"'{FONT_PATH}' 폰트 파일을 찾을 수 없습니다. 텍스트가 이미지에 표시되지 않을 수 있습니다.")
    FONT_PATH = None
    FONT = None

# EasyOCR 리더기 초기화
try:
    reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
except Exception as e:
    print(f"EasyOCR 초기화 중 오류 발생: {e}")
    print("CPU 모드로 다시 시도합니다.")
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

# YOLO 모델 로드
MODEL_PATH = 'license_plate_detector.pt'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' 파일을 찾을 수 없습니다. 터미널에서 'yolo export model=keremberke/yolov8m-license-plate format=pt' 명령을 실행하여 모델을 먼저 다운로드하세요.")
model = YOLO(MODEL_PATH)


def recognize_license_plate(image: np.ndarray) -> (np.ndarray, str, str):
    """
    주어진 이미지에서 자동차 번호판을 탐지, 인식하고, 강조된 박스와 한글 텍스트를 이미지에 그립니다.
    """
    height, width, _ = image.shape
    resolution_text = f"{width} x {height} 픽셀"

    # YOLO 추론을 위해 BGR 형식 사용
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image_bgr)[0]
    detected_texts = []

    # 모든 그리기 작업을 위해 원본 RGB 이미지를 기반으로 Pillow Image 객체 생성
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:
            # --- 박스 스타일 변경 ---
            # 탐지된 번호판 주변에 사각형(Box)을 그립니다.
            # 색상(outline)과 두께(width)를 조절하여 눈에 띄게 만듭니다.
            draw.rectangle(
                [(int(x1), int(y1)), (int(x2), int(y2))],  # 좌표
                outline="red",  # <--- 박스 색상을 'green'에서 'red'로 변경
                width=5         # <--- 박스 두께를 3에서 5로 변경
            )
            # --- ---

            # OCR을 위한 번호판 영역 자르기
            license_plate_crop = image_bgr[int(y1):int(y2), int(x1):int(x2)]
            ocr_result = reader.readtext(license_plate_crop)

            if ocr_result:
                text_parts = [res[1] for res in ocr_result]
                filtered_text_parts = [part.replace(" ", "") for part in text_parts if len(part) > 1]
                text = ' '.join(filtered_text_parts).strip()
                detected_texts.append(text)

                if FONT:
                    # 텍스트 위치 계산 및 그리기
                    text_position = (int(x1) + 5, int(y1) - 40)
                    # 텍스트 배경 사각형 그리기
                    text_bbox = draw.textbbox(text_position, text, font=FONT)
                    padded_bbox = [text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5]
                    draw.rectangle(padded_bbox, fill="red")
                    # 텍스트 그리기 (흰색)
                    draw.text(text_position, text, font=FONT, fill=(255, 255, 255))

    # 최종 결과 이미지를 Numpy 배열로 변환
    output_image = np.array(pil_image)
    recognized_text = ", ".join(detected_texts) if detected_texts else "번호판을 인식하지 못했습니다."
    return output_image, recognized_text, resolution_text


iface = gr.Interface(
    fn=recognize_license_plate,
    inputs=gr.Image(type="numpy", label="자동차 이미지를 업로드하세요"),
    outputs=[
        gr.Image(type="numpy", label="결과 이미지"),
        gr.Textbox(label="인식된 번호판"),
        gr.Textbox(label="입력 이미지 해상도")
    ],
    title="YOLOv8 & EasyOCR 자동차 번호판 인식",
    description="YOLOv8로 자동차 번호판을 탐지하고 EasyOCR로 텍스트를 인식하는 Gradio 앱입니다. 인식된 한글 번호판을.",
)

if __name__ == "__main__":
    iface.launch()