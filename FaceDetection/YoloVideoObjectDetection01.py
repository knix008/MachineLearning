import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os

# Check GPU availability
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# YOLOv8 사전학습 객체 탐지 모델 로드 (자동 다운로드)
model = YOLO("yolov8n.pt")  # 필요에 따라 yolov8s.pt, yolov8m.pt 등으로 변경 가능
model.to(device)  # GPU가 있다면 GPU로 모델을 이동


def detect_objects_in_video(input_video):
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))

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
                # Box 및 라벨 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        out.write(frame)

    cap.release()
    out.release()
    if not isinstance(input_video, str):
        os.remove(video_path)

    return temp_output_path


iface = gr.Interface(
    fn=detect_objects_in_video,
    inputs=gr.Video(label="객체 탐지할 영상을 업로드하세요"),
    outputs=gr.Video(label="탐지 결과 영상"),
    title="YOLOv8 객체 탐지 데모",
    description="YOLOv8로 업로드한 동영상에서 객체를 인식하고 박스를 그려 표시합니다.",
)

if __name__ == "__main__":
    iface.launch()
