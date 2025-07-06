import cv2
import numpy as np
import gradio as gr
import tempfile
import os
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")


def detect_objects_on_video(video_path):
    # 임시 디렉토리에 결과 영상 파일 저장
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"result_{os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_labels = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
            label = (
                results[0].names[cls_id]
                if hasattr(results[0], "names") and cls_id != -1
                else str(cls_id)
            )
            detected_labels.add(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        out.write(frame)

    cap.release()
    out.release()

    # Gradio의 Video 컴포넌트는 mp4 파일 경로를 반환하면 바로 재생이 됩니다.
    labels_text = (
        "\n".join(detected_labels) if detected_labels else "No objects detected."
    )
    return output_path, labels_text


demo = gr.Interface(
    fn=detect_objects_on_video,
    inputs=gr.Video(label="Video Input"),
    outputs=[gr.Video(label="Detected Video"), gr.Textbox(label="Labels")],
    title="YOLOv11 Video Object Detection",
    description="동영상을 업로드하면, YOLOv8로 객체를 검출하여 결과 영상을 보여줍니다. (Replay 지원)",
)

if __name__ == "__main__":
    demo.launch()
