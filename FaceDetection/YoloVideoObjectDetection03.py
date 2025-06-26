import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os
import random

# Load the YOLOv8 model (replace with your own model if available)
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


# 클래스별 색상 미리 지정 (class_id: (B, G, R))
def get_color(idx):
    random.seed(idx)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def detect_objects_in_video_realtime(input_video):
    """
    Detect objects in each frame of the input video using YOLOv8.
    Draw each class with a different colored bounding box.
    Yields processed frames for streaming display in Gradio.
    """
    if isinstance(input_video, str):
        video_path = input_video
    elif hasattr(input_video, "read"):
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(input_video.read())
        temp_input.close()
        video_path = temp_input.name
    else:
        raise ValueError("지원하지 않는 입력 형식입니다.")

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls[i])
                label = (
                    model.model.names[class_id]
                    if hasattr(model.model, "names")
                    else str(class_id)
                )
                confidence = conf[i]
                color = get_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

    cap.release()
    if not isinstance(input_video, str):
        os.remove(video_path)


iface = gr.Interface(
    fn=detect_objects_in_video_realtime,
    inputs=gr.Video(label="객체 탐지할 영상을 업로드하세요"),
    outputs=gr.Image(label="실시간 탐지 결과", streaming=True),
    title="YOLOv8 객체 탐지 실시간 데모",
    description="YOLOv8로 동영상에서 프레임별 객체 인식 결과를 실시간으로 보여줍니다.\n(각 클래스별로 다른 색상으로 박스가 표시됩니다.)",
)

if __name__ == "__main__":
    iface.launch()
