import gradio as gr
import cv2
from ultralytics import YOLO
import os
import datetime
import numpy as np 

# YOLO 모델을 GPU에서 로드
model = YOLO("yolo11n.pt")


def track_video_realtime(video_file, progress=gr.Progress()):
    if video_file is None:
        yield None, None, "비디오 파일이 업로드되지 않았습니다."
        return

    cap = cv2.VideoCapture(video_file.name)
    if not cap.isOpened():
        yield None, None, "비디오 파일을 열 수 없습니다."
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        cap.release()
        yield None, None, "비디오에서 프레임을 읽을 수 없습니다."
        return

    # 결과 비디오 저장 경로
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"yolo_tracking_result_{timestamp}.mp4"
    output_path = os.path.join(os.getcwd(), output_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # GPU를 이용한 YOLO 추적
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # 실시간 미리보기(프레임은 RGB로 변환해서 출력)
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        progress_value = frame_count / total_frames
        status = f"실시간 처리 중... ({frame_count}/{total_frames} 프레임, {progress_value*100:.1f}%)"
        yield img_rgb, None, status

    cap.release()
    out.release()
    yield None, output_path, f"완료! 결과 비디오가 {output_path}에 저장되었습니다."


with gr.Blocks(title="YOLO 실시간 객체 추적 (GPU)") as demo:
    gr.Markdown("# YOLO 객체 추적 시스템 (GPU + 실시간)")
    gr.Markdown(
        "업로드한 비디오에서 YOLO + GPU를 사용해 객체 추적을 실시간으로 미리보고, 처리 후 결과 비디오도 확인할 수 있습니다."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="비디오 파일 업로드", file_types=["video"], file_count="single"
            )
            process_btn = gr.Button("객체 추적 시작", variant="primary")
        with gr.Column():
            frame_output = gr.Image(label="실시간 처리 프레임", interactive=False)
            video_output = gr.Video(
                label="YOLO 추적 결과 비디오", autoplay=True, height=400
            )
            progress_info = gr.Textbox(label="상태", interactive=False)

    # process_btn 클릭 시 실시간 추적 진행
    process_btn.click(
        fn=track_video_realtime,
        inputs=video_input,
        outputs=[frame_output, video_output, progress_info],
        api_name="track_video_realtime",
    )

if __name__ == "__main__":
    demo.launch()
