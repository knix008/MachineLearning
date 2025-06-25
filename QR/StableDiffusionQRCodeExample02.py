import gradio as gr
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from PIL import Image
import qrcode


# 컨트롤넷 및 스테이블 디퓨전 모델 로드
controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
#pipe.enable_xformers_memory_efficient_attention()


# QR 코드 생성 및 이미지 합성 함수
def generate_image_with_qr(
    image,
    text,
    prompt,
    negative_prompt,
    guidance_scale,
    controlnet_conditioning_scale,
    num_inference_steps,
):
    """
    입력 이미지와 텍스트로 QR 코드를 생성하고, 이를 이미지에 자연스럽게 합성합니다.

    Args:
        image (PIL.Image.Image): 원본 이미지
        text (str): QR 코드에 담을 텍스트 (URL 등)
        prompt (str): 이미지 생성을 위한 긍정 프롬프트
        negative_prompt (str): 이미지 생성을 위한 부정 프롬프트
        guidance_scale (float): 프롬프트 충실도
        controlnet_conditioning_scale (float): 컨트롤넷 영향력
        num_inference_steps (int): 추론 스텝 수

    Returns:
        tuple: (최종 합성 이미지, 생성된 QR 코드 이미지)
    """
    # QR 코드 생성
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_code_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_code_image = qr_code_image.resize(image.size)

    # 이미지 생성
    generator = torch.manual_seed(42)
    output_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qr_code_image,
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        width=image.width,
        height=image.height,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        generator=generator,
    ).images[0]

    return output_image, qr_code_image


# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    gr.Markdown("# 이미지에 보이지 않는 QR 코드 삽입 (생성형 AI 활용)")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="원본 이미지")
            qr_text = gr.Textbox(label="QR 코드에 넣을 텍스트 (예: URL)")
            prompt = gr.Textbox(label="긍정 프롬프트", value="a beautiful landscape")
            negative_prompt = gr.Textbox(
                label="부정 프롬프트",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            guidance_scale = gr.Slider(
                minimum=0.0, maximum=20.0, value=7.5, label="Guidance Scale"
            )
            controlnet_conditioning_scale = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=1.5,
                label="ControlNet Conditioning Scale",
            )
            num_inference_steps = gr.Slider(
                minimum=1, maximum=100, value=20, step=1, label="Inference Steps"
            )
            run_button = gr.Button("생성하기")
        with gr.Column():
            output_image = gr.Image(type="pil", label="결과 이미지")
            qr_code_display = gr.Image(type="pil", label="생성된 QR 코드")

    run_button.click(
        fn=generate_image_with_qr,
        inputs=[
            input_image,
            qr_text,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            num_inference_steps,
        ],
        outputs=[output_image, qr_code_display],
    )

demo.launch()
