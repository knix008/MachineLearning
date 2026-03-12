import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# --- 모델 설정 ---
WEIGHTS_DIR = Path(__file__).parent / "weights"

# 모델별 지원 배율
MODEL_SCALES = {
    "espcn":  [2, 3, 4],
    "edsr":   [2, 3, 4],
    "fsrcnn": [2, 3, 4],
    "lapsrn": [2, 4, 8],
}

# 결과 저장 디렉토리 (스크립트와 동일한 디렉토리)
OUTPUT_DIR = Path(__file__).parent


def upscale_image(image_pil, model_name, model_scale, output_format, gr_progress=gr.Progress()):
    if image_pil is None:
        gr.Warning("이미지를 먼저 업로드해 주세요.")
        return None, "", ""

    model_scale = int(model_scale)
    model_path = str(WEIGHTS_DIR / f"{model_name.upper()}_x{model_scale}.pb")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일 '{model_path}'을 찾을 수 없습니다. "
            f"weights/ 디렉토리에 모델 파일을 저장했는지 확인해주세요."
        )

    gr_progress(0.0, desc="전처리 중...")
    print("전처리 중...")

    # PIL 이미지를 OpenCV 형식으로 변환
    image_cv = np.array(image_pil)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # DNN Super Resolution 모델 로드
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(model_name, model_scale)

    # 처리 시간 측정 시작
    gr_progress(0.2, desc="업스케일 중...")
    start_time = time.time()

    with tqdm(total=1, desc=f"{model_name.upper()} x{model_scale} Upscaling", unit="image") as pbar:
        result_cv = sr.upsample(image_cv)
        pbar.update(1)

    elapsed = time.time() - start_time
    elapsed_text = f"처리 시간: {elapsed:.4f}초 ({model_name.upper()} x{model_scale})"
    print(elapsed_text)

    gr_progress(0.85, desc="후처리 중...")
    print("후처리 중...")

    # OpenCV 이미지를 PIL 형식으로 변환
    result_cv = cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_cv)

    # 파일 저장
    gr_progress(0.95, desc="파일 저장 중...")
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    script_name = Path(__file__).stem
    ext = "jpg" if output_format == "JPG" else "png"
    filename = f"{script_name}_{now}_{model_name.upper()}_x{model_scale}.{ext}"
    save_path = OUTPUT_DIR / filename
    if ext == "jpg":
        result_pil.convert("RGB").save(save_path, format="JPEG", quality=95)
    else:
        result_pil.save(save_path)
    print(f"저장 완료: {save_path}")

    gr_progress(1.0, desc="완료!")
    return result_pil, elapsed_text, str(save_path)


# --- Gradio 인터페이스 설정 ---
def update_scale_choices(model_name):
    scales = MODEL_SCALES.get(model_name, [2, 3, 4])
    return gr.Radio(choices=[str(s) for s in scales], value=str(scales[-1]))

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 딥러닝 이미지 업스케일러 (OpenCV DNN Super Resolution)")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="입력 이미지", height=600, sources=["upload", "clipboard"])
            with gr.Accordion("Model Settings", open=True):
                model_name = gr.Dropdown(
                    choices=list(MODEL_SCALES.keys()),
                    value="espcn",
                    label="모델 ★ESPCN 권장 (빠른 속도)",
                    info="ESPCN: 빠름 / FSRCNN: 빠름+고품질 / EDSR: 고품질 / LapSRN: 고배율"
                )
                model_scale = gr.Radio(
                    choices=["2", "3", "4"],
                    value="4",
                    label="업스케일 배율 ★4x 권장",
                    info="LapSRN은 2, 4, 8 지원 / 나머지는 2, 3, 4 지원"
                )
                model_name.change(fn=update_scale_choices, inputs=model_name, outputs=model_scale)
                output_format = gr.Radio(
                    ["JPG", "PNG"],
                    value="JPG",
                    label="저장 파일 형식 ★JPG 권장",
                    info="JPG(권장): 용량 작음, quality 95 / PNG: 무손실"
                )
            submit_button = gr.Button("이미지 확대 실행", variant="primary")
        with gr.Column():
            image_output = gr.Image(type="pil", label="결과 이미지", height=800)
            time_output = gr.Textbox(label="처리 시간", interactive=False)
            save_path_text = gr.Textbox(label="저장 경로", interactive=False)

    submit_button.click(
        fn=upscale_image,
        inputs=[image_input, model_name, model_scale, output_format],
        outputs=[image_output, time_output, save_path_text]
    )

    # 예시 이미지가 있다면 추가
    if os.path.exists("example_image.png"):
        gr.Examples(
            examples=[["example_image.png"]],
            inputs=image_input,
            outputs=[image_output, time_output, save_path_text],
            fn=upscale_image,
            cache_examples=True
        )

# Gradio 앱 실행
if __name__ == "__main__":
    demo.launch(inbrowser=True)
