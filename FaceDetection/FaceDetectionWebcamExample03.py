import gradio as gr
import cv2
import numpy as np

# 얼굴 인식기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    if frame is None:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        # 빨간색, 두꺼운 테두리
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

iface = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="실시간 얼굴 추적 (Gradio + OpenCV)",
    description="웹캠 스트리밍 영상에서 얼굴을 빨간색 두꺼운 사각형으로 실시간 추적합니다."
)

if __name__ == "__main__":
    iface.launch()