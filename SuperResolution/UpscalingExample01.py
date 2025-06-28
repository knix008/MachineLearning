import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import os

# --- 모델 설정 ---
# 사용 가능한 모델: edsr, espcn, fsrcnn, lapsrn
# 모델마다 필요한 스케일(x2, x3, x4)이 다를 수 있습니다.
MODEL_NAME = "edsr"
MODEL_SCALE = 4
MODEL_PATH = f"{MODEL_NAME.upper()}_x{MODEL_SCALE}.pb"
print("> Model name : ", MODEL_PATH)

# Check opencv version
print("> The OpenCV version : ", cv2.__version__)


def upscale_image(image_pil):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. "
            f"https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/{MODEL_PATH} 에서 다운로드하여 "
            "코드와 같은 디렉토리에 저장해주세요."
        )

    # PIL 이미지를 OpenCV 형식으로 변환
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # DNN Super Resolution 모델 로드
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    print("> DNN Super Resolution model created.")
    sr.readModel(MODEL_PATH)
    sr.setModel(MODEL_NAME, MODEL_SCALE)

    # 처리 시간 측정 시작
    start_time = time.time()

    # 이미지 업스케일링 실행
    result_cv = sr.upsample(image_cv)

    # 처리 시간 측정 종료
    end_time = time.time()
    processing_time = f"{end_time - start_time:.2f} 초"

    # OpenCV 이미지를 PIL 형식으로 변환
    result_cv = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_cv)

    return result_pil, processing_time


# --- Gradio 인터페이스 설정 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 딥러닝 이미지 업스케일러 (화질 손실 최소화)
        확대하고 싶은 이미지를 업로드하면 딥러닝 모델(EDSR x4)을 사용하여 화질 손실을 최소화하며 4배 확대합니다.
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

# Gradio 앱 실행
if __name__ == "__main__":
    demo.launch()