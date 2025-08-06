import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import datetime
import gradio as gr
from PIL import Image

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 모델 로딩
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
print("모델을 CPU로 로딩 완료!")

MAX_IMAGE_SIZE = 512  # 최대 이미지 크기
UPSCALE_FACTOR = 4  # 업스케일 배율

def resize_image(image):
    """이미지를 업스케일링하고 최대 크기를 유지합니다."""
    w, h = image.size
    w = (w //16) * 16  # 16의 배수로 조정
    h = (h //16) * 16  # 16의 배수로
    new_w = min(w * UPSCALE_FACTOR, MAX_IMAGE_SIZE)
    new_h = min(h * UPSCALE_FACTOR, MAX_IMAGE_SIZE)
    # 이미지 크기 조정
    image = image.resize((new_w, new_h), Image.LANCZOS)
    return image

def upscale_image(
    input_image,
    guidance_scale,
    num_inference_steps,
    controlnet_conditioning_scale,
):
    if input_image is None:
        return None, "이미지를 업로드하세요."

    # 입력 이미지 크기 조정
    resized_image = resize_image(input_image)

    try:
        image = pipe(
            control_image=resized_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=resized_image.height,
            width=resized_image.width,
        ).images[0]

        filename = f"flux1-dev-controlnet-Upscaler05-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)

        info = f"생성 완료!\n저장 파일: {filename}\n최종 크기: {resized_image.width}x{resized_image.height}\n가이던스 스케일: {guidance_scale}\n추론 스텝: {num_inference_steps}\n컨디셔닝 스케일: {controlnet_conditioning_scale}"

        return image, info
    except Exception as e:
        return None, f"오류 발생: {str(e)}"


with gr.Blocks(title="FLUX.1 ControlNet 업스케일러") as demo:
    gr.Markdown("# 🖼️ FLUX.1 ControlNet 업스케일러")
    gr.Markdown("이미지를 업로드하고, 다양한 설정으로 고해상도 이미지를 생성하세요!")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="입력 이미지",
                type="pil",
                sources=["upload", "clipboard"],
                height=500,
                value="default.jpg",  # 기본 이미지 경로 (예시용)
            )
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=3.5,
                step=0.1,
                label="가이던스 스케일",
                info="프롬프트 준수 정도. 높을수록 프롬프트를 더 정확히 따름.",
            )
            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,
                value=28,
                step=1,
                label="추론 스텝 수",
                info="이미지 생성 단계 수. 높을수록 품질이 향상되지만 생성 시간이 늘어남.",
            )
            conditioning_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.6,
                step=0.05,
                label="컨디셔닝 스케일",
                info="ControlNet의 영향력. 높을수록 입력 이미지에 더 강하게 반영됨.",
            )
            generate_btn = gr.Button(
                "🖼️ 업스케일 이미지 생성", variant="primary", size="lg"
            )

        with gr.Column(scale=1):
            output_image = gr.Image(label="업스케일 결과", type="pil")
            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    generate_btn.click(
        fn=upscale_image,
        inputs=[
            input_image,
            guidance_slider,
            steps_slider,
            conditioning_slider,
        ],
        outputs=[output_image, info_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
