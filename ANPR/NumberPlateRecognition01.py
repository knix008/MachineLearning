import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import os

# 한국어 모델을 사용하는 EasyOCR 리더기 초기화
# gpu=True 설정 시 CUDA가 설치된 환경에서 더 빠른 속도로 처리 가능
reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())

# 사전 학습된 YOLOv8 모델 로드
model = YOLO('license_plate_detector.pt')  # 로컬에 저장된 YOLOv8 모델 파일 경로

def recognize_license_plate(image: np.ndarray) -> (np.ndarray, str):
    """
    주어진 이미지에서 자동차 번호판을 탐지하고 인식합니다.

    Args:
        image (np.ndarray): 입력 이미지 (OpenCV 형식, BGR).

    Returns:
        tuple[np.ndarray, str]: 번호판에 바운딩 박스와 인식된 텍스트가 그려진 이미지와 인식된 텍스트.
    """
    # YOLO 모델을 사용하여 이미지에서 객체 탐지
    results = model(image)[0]

    detected_texts = []

    # 탐지된 각 객체에 대해 반복
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:  # 신뢰도 50% 이상인 경우만 처리
            # 번호판 영역 자르기
            license_plate_crop = image[int(y1):int(y2), int(x1):int(x2)]

            # EasyOCR을 사용하여 번호판 텍스트 인식
            ocr_result = reader.readtext(license_plate_crop)

            if ocr_result:
                # 인식된 텍스트들을 결합
                text_parts = [res[1] for res in ocr_result]
                # 인식률이 낮은 짧은 문자(노이즈) 제거
                filtered_text_parts = [part for part in text_parts if len(part) > 1]
                text = ' '.join(filtered_text_parts)
                detected_texts.append(text)

                # 원본 이미지에 바운딩 박스 및 텍스트 그리기
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    recognized_text = ", ".join(detected_texts) if detected_texts else "번호판을 인식하지 못했습니다."

    return image, recognized_text

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=recognize_license_plate,
    inputs=gr.Image(type="numpy", label="자동차 이미지를 업로드하세요"),
    outputs=[
        gr.Image(type="numpy", label="결과 이미지"),
        gr.Textbox(label="인식된 번호판")
    ],
    title="YOLOv8 & EasyOCR 자동차 번호판 인식",
    description="YOLOv8로 자동차 번호판을 탐지하고 EasyOCR로 텍스트를 인식하는 Gradio 앱입니다. 이미지를 업로드하고 'Submit' 버튼을 누르세요.",
    examples=[
        [os.path.join(os.path.dirname(__file__), "example_car.jpg")]
    ]
)

# Gradio 앱 실행
if __name__ == "__main__":
    iface.launch()