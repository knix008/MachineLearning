import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os

# Load the YOLOv11 model (replace with your own model if available)
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

def detect_objects_in_video_realtime(input_video):
    # Gradio에서 받은 입력이 str(경로)이면 그대로, 아니면 임시 파일로 저장
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
                label = model.names[int(cls[i])]
                confidence = conf[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
        # BGR to RGB 변환 (Gradio Image 컴포넌트는 RGB 사용)
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
    description="YOLOv8로 동영상에서 프레임별 객체 인식 결과를 실시간으로 보여줍니다.",
)

if __name__ == "__main__":
    iface.launch()