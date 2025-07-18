import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
import datetime

# Load the YOLO model
model = YOLO("yolo11n.pt")


def track_video(video_file, progress=gr.Progress()):
    if video_file is None:
        return None
    
    # Gradio File input은 파일 객체이므로, .name으로 경로 추출
    progress(0, desc="비디오 로딩 중...")
    cap = cv2.VideoCapture(video_file.name)
    
    if not cap.isOpened():
        progress(0, desc="비디오 파일을 열 수 없습니다.")
        return None
    
    # 전체 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames == 0:
        progress(0, desc="비디오에서 프레임을 읽을 수 없습니다.")
        cap.release()
        return None

    # 임시 저장용 비디오 코덱 및 파일 (실행 디렉토리에 저장)
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
            
        # YOLO 객체 추적 수행
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        frame_count += 1
        # Progress bar 업데이트 (10프레임마다 업데이트)
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress_value = frame_count / total_frames
            progress(progress_value, desc=f"처리 중... ({frame_count}/{total_frames} 프레임, {progress_value*100:.1f}%)")
    
    cap.release()
    out.release()
    
    progress(1.0, desc="처리 완료! 비디오 재생 준비 중...")
    return output_path


with gr.Blocks(title="YOLO 객체 추적 시스템") as demo:
    gr.Markdown("# YOLO 객체 추적 시스템")
    gr.Markdown("업로드한 비디오에서 YOLO를 사용한 객체 추적 결과를 확인하세요.")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.File(
                label="비디오 파일 업로드", 
                file_types=["video"],
                file_count="single"
            )
            process_btn = gr.Button("객체 추적 시작", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            video_output = gr.Video(
                label="YOLO 추적 결과", 
                autoplay=True,
                height=400
            )
    
    # Progress bar를 별도 컴포넌트로 표시
    progress_info = gr.Textbox(
        label="처리 상태", 
        value="비디오를 업로드하고 '객체 추적 시작' 버튼을 클릭하세요.",
        interactive=False
    )
    
    process_btn.click(
        fn=track_video,
        inputs=video_input,
        outputs=video_output,
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch()
