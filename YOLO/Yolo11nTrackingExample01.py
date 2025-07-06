import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load the YOLO model
model = YOLO("yolo11n.pt")


def track_video(video_file):
    # Gradio File input은 파일 객체이므로, .name으로 경로 추출
    cap = cv2.VideoCapture(video_file.name)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 임시 저장용 비디오 코덱 및 파일
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    cap.release()
    out.release()
    os.close(temp_fd)

    return temp_path


demo = gr.Interface(
    fn=track_video,
    inputs=gr.File(label="비디오 업로드"),
    outputs=gr.Video(label="YOLO Tracking 결과"),
    title="YOLO Tracking with Gradio",
    description="업로드한 비디오에서 객체 추적 결과를 보여줍니다.",
)

if __name__ == "__main__":
    demo.launch()
