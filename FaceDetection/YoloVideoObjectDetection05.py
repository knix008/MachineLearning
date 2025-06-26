import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os
import random
import torch

# CUDA 사용 가능하면 GPU에 올림
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")
model.to(device)  # 모델을 GPU로 이동


def get_color(idx):
    random.seed(idx)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def detect_objects_in_video_realtime(input_video):
    """
    Detect objects in each frame of the input video using YOLOv8 and OpenCV CUDA.
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
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV: 프레임을 GPU로 올려서 처리
        if use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)

            # YOLO 추론을 위해 CPU 메모리로 복사
            frame_for_yolo = gpu_frame.download()
        else:
            frame_for_yolo = frame

        # YOLO 추론 (GPU)
        results = model(frame_for_yolo, device=device)
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
                if use_cuda:
                    # GPU에서 rectangle, putText는 지원이 제한적이므로 CPU에서 그림
                    cv2.rectangle(frame_for_yolo, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame_for_yolo,
                        f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                else:
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

        # RGB 변환도 GPU 사용
        if use_cuda:
            gpu_final = cv2.cuda_GpuMat()
            gpu_final.upload(frame_for_yolo)
            gpu_rgb = cv2.cuda.cvtColor(gpu_final, cv2.COLOR_BGR2RGB)
            frame_rgb = gpu_rgb.download()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

    cap.release()
    if not isinstance(input_video, str):
        os.remove(video_path)


iface = gr.Interface(
    fn=detect_objects_in_video_realtime,
    inputs=gr.Video(label="객체 탐지할 영상을 업로드하세요"),
    outputs=gr.Image(label="실시간 탐지 결과", streaming=True),
    title="YOLOv8 객체 탐지 실시간 데모 (GPU+OpenCV CUDA)",
    description="YOLOv8로 동영상에서 프레임별 객체 인식 결과를 실시간으로 보여줍니다.\n(각 클래스별로 다른 색상으로 박스가 표시됩니다.)\nYOLO와 OpenCV 모두 GPU(CUDA) 가속을 지원합니다.",
)

if __name__ == "__main__":
    iface.launch()
