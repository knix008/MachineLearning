import gradio as gr
import cv2
from ultralytics import YOLO
import os

# YOLO 얼굴탐지 모델 파일명 (예: yolov8x-face-lindevs.pt)
yolo_model_path = "yolov8x-face-lindevs.pt"
model = YOLO(yolo_model_path)
DEFAULT_SAVE_FILENAME = "output_face_detected.mp4"

def detect_faces_and_save_with_progress(input_video, progress=gr.Progress(track_tqdm=True)):
    save_path = os.path.join(os.getcwd(), DEFAULT_SAVE_FILENAME)
    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # 'mp4v' for .mp4 files
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    processed = 0
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
        out.write(annotated_frame)
        processed += 1
        if total_frames > 0:
            progress(processed / total_frames)
    cap.release()
    out.release()
    # 결과 동영상 경로 반환
    return save_path

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # YOLO 얼굴 인식 (동영상 결과 입력과 같은 행에서 재생)
        1. 동영상을 업로드하고 실행 버튼을 누르세요.
        2. 진행상태(progress bar)가 보이고, 결과 동영상은 입력 영상과 같은 행에서 재생됩니다.
        """
    )
    with gr.Row():
        input_video = gr.Video(label="업로드 동영상")
        result_video = gr.Video(label="YOLO 얼굴 인식 결과 (재생)")

    run_btn = gr.Button("얼굴 인식 실행 및 결과 저장/재생")

    run_btn.click(
        detect_faces_and_save_with_progress,
        inputs=input_video,
        outputs=result_video
    )

if __name__ == "__main__":
    demo.launch()