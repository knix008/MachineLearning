import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import time
import datetime
import gc
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def get_model():
    """
    업스케일에 사용할 RRDBNet 모델과 관련 정보를 반환합니다.
    Returns:
        model: 업스케일용 네트워크 객체
        model_path: 가중치 파일 경로
        netscale: 업스케일 배수(4)
        dni_weight: DNI 가중치(혼합 비율, None 또는 리스트)
    """
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
    dni_weight = None
    # Set device for Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model, model_path, dni_weight, device


def enhance_image(
    input_img,   # 업스케일할 입력 이미지 (PIL.Image)
    upscale,    # 업스케일 배수 (1~4)
    tile,        # 타일 크기 (메모리 부족시 분할 처리)
    tile_pad,    # 타일 경계 패딩
    pre_pad,     # 전체 이미지 패딩
    fp32,        # FP32 연산 사용 여부
    ext,         # 저장 확장자 (png/jpg)
):
    """
    입력 이미지를 업스케일하고, 결과 이미지를 반환합니다.
    Args:
        input_img (PIL.Image): 업스케일할 이미지
        upscale (int): 업스케일 배수
        tile (int): 타일 크기 (0은 전체 처리)
        tile_pad (int): 타일 경계 패딩
        pre_pad (int): 전체 이미지 패딩
        fp32 (bool): FP32 연산 사용 여부
        ext (str): 저장 확장자
    Returns:
        PIL.Image, str: 업스케일된 이미지와 상태 메시지
    """
    start_time = time.time()
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)
    model, model_path, dni_weight, device = get_model()
    model = model.to(device)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=None,  # Keep None for MPS
        device=device, # Pass device explicitly
    )
    try:
        output, _ = upsampler.enhance(img, outscale=upscale)
    except Exception as e:
        # 메모리 해제
        del img, model, upsampler
        gc.collect()
        return None, f"Error: {str(e)}"

    # 메모리 해제
    del img, model, upsampler
    gc.collect()

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(output)
    ext_to_use = ext
    filename = f"RealESRGAN_Example02_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{ext_to_use}"
    output.save(filename)
    elapsed = time.time() - start_time
    print(f"The Elapsed Time : {elapsed:.2f} seconds")
    return output, f"완료! 처리 시간: {elapsed:.2f}초"


with gr.Blocks() as demo:
    gr.Markdown("# Real-ESRGAN 업스케일러 (Gradio 데모)")
    gr.Markdown("""
    **사용법 안내**
    1. 이미지를 업로드하거나 붙여넣으세요.
    2. 업스케일 배수, 타일 크기 등 옵션을 조정하세요.
    3. [업스케일 실행] 버튼을 누르면 결과 이미지를 확인할 수 있습니다.
    """)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="입력 이미지 (업스케일할 원본)",
                type="pil",
                height=500,
                value="default.png",
            )
            upscale = gr.Slider(
                minimum=1,
                maximum=4,
                value=4,
                step=1,
                label="업스케일 배수 (이미지를 몇 배로 확대할지)",
                info="이미지를 몇 배로 확대할지 설정합니다. (1~4)",
            )
            tile = gr.Slider(
                minimum=4,
                maximum=512,
                value=8,  # 기본값을 더 작게 : 16에서 변경
                step=4,
                label="Tile 크기 (메모리 부족시 분할 처리)",
                info="이미지를 분할 처리할 타일 크기입니다. 메모리 부족 시 값을 줄이세요. 0은 전체 처리.",
            )
            tile_pad = gr.Number(
                minimum=0,
                maximum=512,
                value=10,
                label="Tile padding (타일 경계 패딩)",
                info="타일 경계에 추가로 패딩을 줄 픽셀 수입니다.",
            )
            pre_pad = gr.Number(
                minimum=0,
                maximum=512,
                value=0,
                label="Pre padding (전체 이미지 패딩)",
                info="입력 이미지 전체에 추가로 패딩을 줄 픽셀 수입니다.",
            )
            fp32 = gr.Checkbox(
                label="FP32 모드 사용 (고정소수점 연산)",
                value=False,  # 기본값을 False로
                info="체크 시 FP32(고정소수점) 연산을 사용합니다. (메모리 여유가 많을 때 권장)",
            )
            ext = gr.Radio(
                choices=["png", "jpg"],
                value="png",
                label="저장 확장자 (결과 파일 형식)",
                info="결과 이미지를 저장할 파일 형식입니다.",
            )
            btn = gr.Button("업스케일 실행")
        with gr.Column():
            output_img = gr.Image(label="결과 이미지 (업스케일 결과)")
            status = gr.Textbox(label="상태 메시지")

    btn.click(
        enhance_image,
        inputs=[input_img, upscale, tile, tile_pad, pre_pad, fp32, ext],
        outputs=[output_img, status],
    )

if __name__ == "__main__":
    print(torch.backends.mps.is_available())
    demo.launch(share=False, inbrowser=True)
