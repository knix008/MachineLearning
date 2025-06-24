import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os

# GPU 사용 여부 확인 및 모델 GPU로 이동
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
model = YOLO("yolo11n.pt")
model.to(device)

def detect_objects_in_video_realtime(input_video):
    # 입력이 파일 경로인지 파일 객체인지 구분
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

        # GPU에서 추론 (Ultralytics YOLO는 device 파라미터 지원)
        results = model(frame, device=device)
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                label = model.names[int(cls[i])] if hasattr(model, "names") else str(int(cls[i]))
                confidence = conf[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
        # RGB 변환 및 yield (프레임 단위로 실시간 스트리밍)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb

    cap.release()
    if not isinstance(input_video, str):
        os.remove(video_path)

iface = gr.Interface(
    fn=detect_objects_in_video_realtime,
    inputs=gr.Video(label="객체 탐지할 영상을 업로드하세요"),
    outputs=gr.Image(label="실시간 탐지 결과", streaming=True),
    title="YOLOv8 객체 탐지 실시간 데모 (GPU 지원)",
    description="YOLOv8과 GPU를 활용해 동영상에서 프레임별 객체 인식 결과를 실시간으로 보여줍니다.",
)

if __name__ == "__main__":
    iface.launch()