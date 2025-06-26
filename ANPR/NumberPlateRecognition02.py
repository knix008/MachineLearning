import gradio as gr
import torch
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np
import os

# 한국어 모델을 사용하는 EasyOCR 리더기 초기화
# gpu=True 설정 시 CUDA가 설치된 환경에서 더 빠른 속도로 처리 가능
try:
    reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
except Exception as e:
    print(f"EasyOCR 초기화 중 오류 발생: {e}")
    print("CPU 모드로 다시 시도합니다.")
    reader = easyocr.Reader(['ko', 'en'], gpu=False)


# 1. 모델 파일을 먼저 'yolo export' 명령어로 다운로드해야 합니다.
# 터미널 명령어: yolo export model=keremberke/yolov8m-license-plate format=pt
# 2. 다운로드된 로컬 모델 파일('yolov8m-license-plate.pt')을 로드합니다.
MODEL_PATH = 'license_plate_detector.pt'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' 파일을 찾을 수 없습니다.")

model = YOLO(MODEL_PATH)

def recognize_license_plate(image: np.ndarray) -> (np.ndarray, str, str):
    # ====================================================================
    # 변경점 1: 입력 이미지의 해상도 확인
    # ====================================================================
    height, width, _ = image.shape
    resolution_text = f"{width} x {height} 픽셀"
    # ====================================================================

    # Gradio는 이미지를 RGB로 로드하므로, OpenCV에서 사용하기 위해 BGR로 변환
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # YOLO 모델을 사용하여 이미지에서 객체 탐지
    results = model(image_bgr)[0]

    detected_texts = []

    # 탐지된 각 객체에 대해 반복
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0.5:  # 신뢰도 50% 이상인 경우만 처리
            # 번호판 영역 자르기
            license_plate_crop = image_bgr[int(y1):int(y2), int(x1):int(x2)]

            # EasyOCR은 BGR 이미지를 입력받음
            ocr_result = reader.readtext(license_plate_crop)

            if ocr_result:
                # 인식된 텍스트들을 결합
                text_parts = [res[1] for res in ocr_result]
                # 인식률이 낮은 짧은 문자(노이즈) 제거
                filtered_text_parts = [part for part in text_parts if len(part) > 1]
                text = ' '.join(filtered_text_parts).strip()
                detected_texts.append(text)

                # 원본 이미지(BGR)에 바운딩 박스 및 텍스트 그리기
                cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 텍스트 배경을 위한 사각형 계산
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(image_bgr, (int(x1), int(y1) - text_height - 15), (int(x1) + text_width, int(y1) - 5), (0, 255, 0), -1)
                # 텍스트 그리기 (흰색)
                cv2.putText(image_bgr, text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    recognized_text = ", ".join(detected_texts) if detected_texts else "번호판을 인식하지 못했습니다."

    # 최종 출력을 위해 다시 RGB로 변환
    output_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ====================================================================
    # 변경점 2: 해상도 텍스트를 결과값에 추가하여 반환
    # ====================================================================
    return output_image, recognized_text, resolution_text
    # ====================================================================

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=recognize_license_plate,
    inputs=gr.Image(type="numpy", label="자동차 이미지를 업로드하세요"),
    # ====================================================================
    # 변경점 3: 해상도를 표시할 출력 컴포넌트 추가
    # ====================================================================
    outputs=[
        gr.Image(type="numpy", label="결과 이미지"),
        gr.Textbox(label="인식된 번호판"),
        gr.Textbox(label="입력 이미지 해상도")
    ],
    # ====================================================================
    title="YOLOv8 & EasyOCR 자동차 번호판 인식 (해상도 표시)",
    description="YOLOv8로 자동차 번호판을 탐지하고 EasyOCR로 텍스트를 인식하는 Gradio 앱입니다. 입력된 이미지의 해상도도 함께 표시됩니다.",
)

# Gradio 앱 실행
if __name__ == "__main__":
    iface.launch()