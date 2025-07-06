import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime
import torch

# CUDA 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"

# Load the YOLO model on GPU if available
model = YOLO("yolo11n.pt").to(DEVICE)


def track_video(video_file, progress=gr.Progress()):
    if video_file is None:
        return None

    if not USE_CUDA:
        progress(0, desc="⚠️ 현재 시스템에 GPU(CUDA)가 없습니다. CPU로 실행됩니다.")
    else:
        progress(0, desc="GPU(CUDA)에서 실행 중...")

    cap = cv2.VideoCapture(video_file.name)
    if not cap.isOpened():
        progress(0, desc="비디오 파일을 열 수 없습니다.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        progress(0, desc="비디오에서 프레임을 읽을 수 없습니다.")
        cap.release()
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"yolo_tracking_result_{timestamp}.mp4"
    output_path = os.path.join(os.getcwd(), output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    progress(0, desc=f"객체 추적 처리 시작... (총 {total_frames} 프레임)")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # GPU로 추론(device에서 실행)
        results = model.track(frame, persist=True, device=DEVICE)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress_value = frame_count / total_frames
            progress(
                progress_value,
                desc=f"처리 중... ({frame_count}/{total_frames} 프레임, {progress_value*100:.1f}%)",
            )

    cap.release()
    out.release()

    progress(1.0, desc="처리 완료! 비디오 재생 준비 중...")
    return output_path


def replay_video(output_path):
    # 단순히 video_output에 같은 파일 경로를 다시 반환하면 Gradio가 비디오를 새로 로드해 재생함
    return output_path


with gr.Blocks(title="YOLO 객체 추적 시스템") as demo:
    gr.Markdown("# YOLO 객체 추적 시스템")
    gr.Markdown("업로드한 비디오에서 YOLO를 사용한 객체 추적 결과를 확인하세요.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(
                label="비디오 파일 업로드", file_types=["video"], file_count="single"
            )
            process_btn = gr.Button("객체 추적 시작", variant="primary", size="lg")
        with gr.Column(scale=2):
            video_output = gr.Video(label="YOLO 추적 결과", autoplay=True, height=400)
            # 다시 재생 버튼 추가
            replay_btn = gr.Button("결과 비디오 다시 재생", variant="secondary")

    progress_info = gr.Textbox(
        label="처리 상태",
        value="비디오를 업로드하고 '객체 추적 시작' 버튼을 클릭하세요.",
        interactive=False,
    )

    # 객체 추적 버튼 클릭 시
    process_btn.click(
        fn=track_video, inputs=video_input, outputs=video_output, show_progress=True
    ).then(
        fn=lambda _: "처리 완료! 결과 비디오를 재생할 수 있습니다.",
        inputs=None,
        outputs=progress_info,
    )

    # 다시 재생 버튼 클릭 시
    replay_btn.click(fn=replay_video, inputs=video_output, outputs=video_output)

if __name__ == "__main__":
    demo.launch()
