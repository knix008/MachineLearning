import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import time
import datetime

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


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
    dni_weight = None  # 필요시 [1.0] 또는 [0.7, 0.3] 등으로 변경
    return model, model_path, 4, dni_weight


def enhance_image(
    input_img,
    outscale,
    tile,
    tile_pad,
    pre_pad,
    fp32,
    ext,
):
    start_time = time.time()
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    model, model_path, netscale, dni_weight = get_model()
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
    output = Image.fromarray(output)
    # 확장자 처리: 항상 사용자가 선택한 ext 사용
    ext_to_use = ext
    filename = f"RealESRGAN_Example02_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{ext_to_use}"
    output.save(filename)

    elapsed = time.time() - start_time
    return output, f"완료! 처리 시간: {elapsed:.2f}초"


with gr.Blocks() as demo:
    gr.Markdown("# Real-ESRGAN 업스케일러 (Gradio 데모)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="입력 이미지",
                type="pil",
                height=500,
                value="default.jpg",  # 기본 이미지 경로 (예시용)
            )
            outscale = gr.Slider(
                1,
                4,
                value=4,
                step=1,
                label="업스케일 배수",
                info="이미지를 몇 배로 확대할지 설정합니다. (1~4)",
            )
            tile = gr.Slider(
                0,
                512,
                value=0,
                step=16,
                label="Tile 크기 (메모리 부족시 조정)",
                info="이미지를 분할 처리할 타일 크기입니다. 메모리 부족 시 값을 줄이세요. 0은 전체 처리.",
            )
            tile_pad = gr.Number(
                value=10,
                label="Tile padding",
                info="타일 경계에 추가로 패딩을 줄 픽셀 수입니다.",
            )
            pre_pad = gr.Number(
                value=0,
                label="Pre padding",
                info="입력 이미지 전체에 추가로 패딩을 줄 픽셀 수입니다.",
            )
            fp32 = gr.Checkbox(
                label="FP32 모드 사용",
                value=True,
                info="체크 시 FP32(고정소수점) 연산을 사용합니다. (메모리 여유가 많을 때 권장)",
            )
            ext = gr.Radio(
                choices=["png", "jpg"],
                value="png",
                label="저장 확장자",
                info="결과 이미지를 저장할 파일 형식입니다.",
            )
            btn = gr.Button("업스케일 실행")
        with gr.Column():
            output_img = gr.Image(label="결과 이미지")
            status = gr.Textbox(label="상태")

    btn.click(
        enhance_image,
        inputs=[input_img, outscale, tile, tile_pad, pre_pad, fp32, ext],
        outputs=[output_img, status],
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
