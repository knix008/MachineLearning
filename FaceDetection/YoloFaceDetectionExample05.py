import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

# 얼굴탐지 YOLO 모델 (직접 다운로드 필요, 예시: yolov8x-face-lindevs.pt)
yolo_model_path = "yolov8x-face-lindevs.pt"  # 얼굴 detection 전용 weight
model = YOLO(yolo_model_path)


def detect_faces_in_video(video):
    cap = cv2.VideoCapture(video)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 24  # fallback

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
        frames.append(annotated_frame)
    cap.release()

    if len(frames) == 0:
        return None, None

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_out = temp_file.name
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()

    # 결과 파일 경로와 다운로드용 파일 경로 모두 반환
    return temp_out, temp_out


iface = gr.Interface(
    fn=detect_faces_in_video,
    inputs=gr.Video(label="업로드 동영상"),
    outputs=[
        gr.Video(label="YOLO 얼굴 인식 결과 (재생)"),
        gr.File(label="YOLO 결과 동영상 파일 다운로드"),
    ],
    title="YOLO 얼굴 인식 (동영상 파일)",
    description="YOLO 얼굴 탐지 모델을 이용한 동영상 얼굴 인식 데모입니다. 동영상 파일 내 얼굴만 인식해 박스를 그려줍니다. 결과를 바로 재생하고 파일로도 다운로드할 수 있습니다.",
)

if __name__ == "__main__":
    iface.launch()
