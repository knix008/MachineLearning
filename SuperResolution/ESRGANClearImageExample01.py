import gradio as gr
import cv2
import numpy as np
import os
import time

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 샤프닝 함수 (선명화)
def sharpen_image(img, strength=1.0):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5 + strength, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def get_model():
    # 로컬에 다운로드 받은 모델만 사용 (weights/RealESRGAN_x4plus.pth)
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

def enhance_image(
    input_img,
    outscale,
    tile,
    tile_pad,
    pre_pad,
    fp32,
    sharpen_strength,
):
    start_time = time.time()
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    try:
        model, model_path, netscale = get_model()
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
    except Exception as e:
        return None, f"Error: {str(e)}"
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    elapsed = time.time() - start_time
    return output, f"완료! 처리 시간: {elapsed:.2f}초"

with gr.Blocks() as demo:
    gr.Markdown("# Real-ESRGAN 업스케일러 (샤프닝 포함, Gradio 데모 - 로컬모델 사용)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="입력 이미지", type="pil")
            outscale = gr.Slider(1, 4, value=4, step=0.1, label="업스케일 배수")
            tile = gr.Slider(
                0, 512, value=0, step=16, label="Tile 크기 (메모리 부족시 조정)"
            )
            tile_pad = gr.Number(value=10, label="Tile padding")
            pre_pad = gr.Number(value=0, label="Pre padding")
            fp32 = gr.Checkbox(label="FP32 모드 사용", value=False)
            sharpen_strength = gr.Slider(
                0.0, 3.0, value=1.0, step=0.1, label="샤프닝 강도"
            )
            btn = gr.Button("업스케일 + 샤프닝 실행")
        with gr.Column():
            output_img = gr.Image(label="결과 이미지")
            status = gr.Textbox(label="상태")

    btn.click(
        enhance_image,
        inputs=[
            input_img,
            outscale,
            tile,
            tile_pad,
            pre_pad,
            fp32,
            sharpen_strength,
        ],
        outputs=[output_img, status],
    )

if __name__ == "__main__":
    demo.launch()