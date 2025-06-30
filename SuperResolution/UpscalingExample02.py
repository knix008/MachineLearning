import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import os

# --- 모델 설정 (수정된 부분) ---
# 모델 이름을 'espcn'으로 변경하고, 파일 경로도 새 모델 파일 이름으로 지정합니다.
MODEL_NAME = "espcn"
MODEL_SCALE = 4
MODEL_PATH = f"ESPCN_x{MODEL_SCALE}.pb" # 모델 파일 이름: ESPCN_x4.pb


def upscale_image(image_pil):
    if not image_pil:
        raise gr.Error("먼저 이미지를 업로드해주세요!")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. "
            f"코드와 같은 디렉토리에 모델 파일을 저장했는지 확인해주세요."
        )

    # PIL 이미지를 OpenCV 형식으로 변환
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # DNN Super Resolution 모델 로드
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_PATH)
    sr.setModel(MODEL_NAME, MODEL_SCALE)

    # 처리 시간 측정 시작
    start_time = time.time()

    # 이미지 업스케일링 실행
    result_cv = sr.upsample(image_cv)

    # 처리 시간 측정 종료
    end_time = time.time()
    processing_time = f"{end_time - start_time:.4f} 초 ({MODEL_NAME.upper()})" # 모델 이름 표시

    # OpenCV 이미지를 PIL 형식으로 변환
    result_cv = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_cv)

    return result_pil, processing_time


# --- Gradio 인터페이스 설정 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 딥러닝 이미지 업스케일러 (모델: {MODEL_NAME.upper()})
        확대하고 싶은 이미지를 업로드하면 **{MODEL_NAME.upper()}** 모델을 사용하여 4배 확대합니다.
        이 모델은 빠른 처리 속도가 특징입니다.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="입력 이미지")
            submit_button = gr.Button("이미지 확대 실행", variant="primary")
        with gr.Column():
            image_output = gr.Image(type="pil", label="결과 이미지")
            time_output = gr.Textbox(label="처리 시간")

    submit_button.click(
        fn=upscale_image,
        inputs=image_input,
        outputs=[image_output, time_output]
    )

    # 예시 이미지가 있다면 추가
    if os.path.exists("example_image.png"):
        gr.Examples(
            examples=[["example_image.png"]],
            inputs=image_input,
            outputs=[image_output, time_output],
            fn=upscale_image,
            cache_examples=True
        )

# Gradio 앱 실행
if __name__ == "__main__":
    demo.launch()