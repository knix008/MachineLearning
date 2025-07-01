import gradio as gr
import cv2
import numpy as np
import os
import time

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def get_model(model_name, denoise_strength=0.5):
    model_name = model_name.split(".")[0]
    netscale = 4
    dni_weight = None

    if model_name == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "RealESRGAN_x2plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        netscale = 2
    elif model_name == "realesr-animevideov3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    elif model_name == "realesr-general-x4v3":
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        wdn_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
    else:
        raise ValueError("Unknown model name")

    os.makedirs("weights", exist_ok=True)
    if model_name == "realesr-general-x4v3":
        model_path = [
            load_file_from_url(url=wdn_url, model_dir="weights", progress=True),
            load_file_from_url(url=url, model_dir="weights", progress=True),
        ]
        if denoise_strength != 1:
            dni_weight = [denoise_strength, 1 - denoise_strength]
    else:
        model_path = load_file_from_url(url=url, model_dir="weights", progress=True)

    return model, model_path, netscale, dni_weight


def enhance_image(
    input_img,
    model_name,
    denoise_strength,
    outscale,
    tile,
    tile_pad,
    pre_pad,
    fp32,
    ext,
):
    start_time = time.time()
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    model, model_path, netscale, dni_weight = get_model(model_name, denoise_strength)
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=None,
    )
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except Exception as e:
        return None, f"Error: {str(e)}"
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    elapsed = time.time() - start_time
    return output, f"완료! 처리 시간: {elapsed:.2f}초"


with gr.Blocks() as demo:
    gr.Markdown("# Real-ESRGAN 업스케일러 (Gradio 데모)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="입력 이미지", type="pil")
            model_name = gr.Dropdown(
                choices=[
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRGAN_x2plus",
                    "realesr-animevideov3",
                    "realesr-general-x4v3",
                ],
                value="RealESRGAN_x4plus",
                label="모델 선택",
            )
            denoise_strength = gr.Slider(
                0,
                1,
                value=0.5,
                step=0.05,
                label="Denoise Strength (realesr-general-x4v3 한정)",
            )
            outscale = gr.Slider(1, 4, value=4, step=0.1, label="업스케일 배수")
            tile = gr.Slider(
                0, 512, value=0, step=16, label="Tile 크기 (메모리 부족시 조정)"
            )
            tile_pad = gr.Number(value=10, label="Tile padding")
            pre_pad = gr.Number(value=0, label="Pre padding")
            fp32 = gr.Checkbox(label="FP32 모드 사용", value=False)
            ext = gr.Radio(
                choices=["auto", "png", "jpg"], value="auto", label="저장 확장자"
            )
            btn = gr.Button("업스케일 실행")
        with gr.Column():
            output_img = gr.Image(label="결과 이미지")
            status = gr.Textbox(label="상태")

    btn.click(
        enhance_image,
        inputs=[
            input_img,
            model_name,
            denoise_strength,
            outscale,
            tile,
            tile_pad,
            pre_pad,
            fp32,
            ext,
        ],
        outputs=[output_img, status],
    )

if __name__ == "__main__":
    demo.launch()
