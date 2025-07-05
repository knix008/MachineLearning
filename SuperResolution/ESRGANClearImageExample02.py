import gradio as gr
import cv2
import numpy as np
import os
import time
import tempfile
from PIL import Image

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def sharpen_image(img, strength=1.0):
    kernel = np.array([[0, -1, 0], [-1, 5 + strength, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened


def get_model():
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    model_path = os.path.join("weights", "RealESRGAN_x4plus.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    return model, model_path, 4


def enhance_image(input_img, sharpen_strength):
    start_time = time.time()
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    try:
        model, model_path, netscale = get_model()
        outscale = 4
        tile = 0
        tile_pad = 10
        pre_pad = 0
        fp32 = False

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=not fp32,
            gpu_id=None,
        )
        output, _ = upsampler.enhance(img, outscale=outscale)
        output = sharpen_image(output, sharpen_strength)

        # 결과 이미지를 4배로 축소
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        h, w = output_rgb.shape[:2]
        resized = cv2.resize(output_rgb, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        pil_resized = Image.fromarray(resized)
        pil_full = Image.fromarray(output_rgb)

        # 임시 파일로 저장 (다운로드용)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            pil_full.save(f, format="PNG")
            temp_file_path = f.name

    except Exception as e:
        return None, None, f"Error: {str(e)}"
    elapsed = time.time() - start_time
    return pil_resized, temp_file_path, f"완료! 처리 시간: {elapsed:.2f}초"


with gr.Blocks() as demo:
    gr.Markdown(
        "# Real-ESRGAN 업스케일러 (업스케일 x4 고정, 샤프닝 조절, 4배 축소 결과 제공, Gradio 데모 - 로컬모델 사용)"
    )
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="입력 이미지", type="pil")
            sharpen_strength = gr.Slider(
                0.0, 3.0, value=1.0, step=0.1, label="샤프닝 강도"
            )
            btn = gr.Button("업스케일 + 샤프닝 실행")
        with gr.Column():
            output_img = gr.Image(label="4배 축소 결과 이미지")
            download = gr.File(label="업스케일(원본 크기) 다운로드")
            status = gr.Textbox(label="상태")

    btn.click(
        enhance_image,
        inputs=[
            input_img,
            sharpen_strength,
        ],
        outputs=[output_img, download, status],
    )

if __name__ == "__main__":
    demo.launch()
