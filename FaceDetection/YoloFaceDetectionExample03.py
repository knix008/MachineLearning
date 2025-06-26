import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# 얼굴탐지 YOLO 모델 (직접 다운로드 필요, 예시: yolov8x-face-lindevs.pt)
yolo_model_path = "yolov8x-face-lindevs.pt"  # 얼굴 detection 전용 weight
model = YOLO(yolo_model_path)

def detect_faces_generator(video):
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = frame.copy()
        if results and len(results) > 0:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    # 얼굴 클래스만 필터링 (얼굴 전용 모델은 보통 0번만 존재)
                    if hasattr(box, "cls") and int(box.cls[0]) != 0:
                        continue
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"Face {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        yield annotated_frame
    cap.release()

iface = gr.Interface(
    fn=detect_faces_generator,
    inputs=gr.Video(label="업로드 동영상"),
    outputs=gr.Image(label="YOLO 얼굴 인식 결과"),
    title="YOLO 얼굴 인식 (동영상 파일, 프레임별 실시간 표시)",
    description="YOLO 얼굴 탐지 모델을 이용하여 동영상의 각 프레임별로 인식 결과를 실시간으로 보여줍니다.",
    flagging_mode="never",
    live=True,
)

if __name__ == "__main__":
    iface.launch()