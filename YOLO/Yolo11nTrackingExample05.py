import gradio as gr
import cv2
from ultralytics import YOLO
import os
import datetime

# GPU에서 모델 로드
model = YOLO("yolo11n.pt")


def track_video_progress(video_file, progress=gr.Progress()):
    if video_file is None:
        return "비디오 파일이 업로드되지 않았습니다."

    cap = cv2.VideoCapture(video_file.name)
    if not cap.isOpened():
        return "비디오 파일을 열 수 없습니다."

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        cap.release()
        return "비디오에서 프레임을 읽을 수 없습니다."

    # 결과 비디오 저장 경로 (현재 디렉토리)
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

        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress(
                frame_count / total_frames,
                desc=f"진행률: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)",
            )

    cap.release()
    out.release()
    progress(1.0, desc=f"완료! 결과 파일: {output_filename}")

    return f"완료! 결과 파일이 현재 디렉토리에 저장되었습니다: {output_filename}"


with gr.Blocks(title="YOLO 11 객체 추적 시스템") as demo:
    gr.Markdown("# YOLO 11 객체 추적")
    gr.Markdown(
        "업로드한 비디오에서 YOLO를 사용하여 객체 추적을 수행하고, 결과 파일은 현재 디렉토리에 저장됩니다."
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.File(
                label="비디오 파일 업로드", file_types=["video"], file_count="single"
            )
            process_btn = gr.Button("객체 추적 시작", variant="primary")
        with gr.Column():
            progress_info = gr.Textbox(label="처리 상태", interactive=False)

    process_btn.click(
        fn=track_video_progress,
        inputs=video_input,
        outputs=progress_info,
    )

if __name__ == "__main__":
    demo.launch()
