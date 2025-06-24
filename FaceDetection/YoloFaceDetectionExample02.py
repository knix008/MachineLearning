import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

# 얼굴탐지 YOLO 모델 (직접 다운로드 필요, 예시: yolov8n-face.pt)
yolo_model_path = "yolov8x-face-lindevs.pt"  # 얼굴 detection 전용 weight
model = YOLO(yolo_model_path)

def detect_faces_in_video(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = frame.copy()
        if results and len(results) > 0:
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    # 얼굴 클래스만 필터링
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
    # 그라디오에서 동영상 반환을 위해, 프레임 리스트를 mp4로 인코딩
    if len(frames) == 0:
        return None
    # 임시 파일에 저장 후 반환
    temp_out = "output_face_detected.mp4"
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out, fourcc, 24, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    return temp_out


iface = gr.Interface(
    fn=detect_faces_in_video,
    inputs=gr.Video(label="업로드 동영상"),
    outputs=gr.Video(label="YOLO 얼굴 인식 결과"),
    title="YOLO 얼굴 인식 (동영상)",
    description="YOLO 얼굴 탐지 모델을 이용한 동영상 얼굴 인식 데모입니다. 동영상 내 얼굴만 인식해 박스를 그려줍니다.",
)

if __name__ == "__main__":
    iface.launch()
