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
upscale_pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", controlnet=controlnet, torch_dtype=torch.bfloat16
)

upscale_pipe.enable_model_cpu_offload()
upscale_pipe.enable_sequential_cpu_offload()
upscale_pipe.enable_attention_slicing(1)
print("모델을 CPU로 로딩 완료!")

MAX_IMAGE_SIZE = 512

def resize_image(input_image, upscale_factor):
    w, h = input_image.size
    # MAX_IMAGE_SIZE 이하로, 가로세로 비율 유지하며 리사이즈
    scale = min(MAX_IMAGE_SIZE / w, MAX_IMAGE_SIZE / h, 1.0)
    resized_w = int(w * scale)
    resized_h = int(h * scale)
    input_image = input_image.resize((resized_w, resized_h), Image.LANCZOS)
    #input_image.save("input_image_resized.png")
    # Upscaler
    upscaled_w = resized_w * upscale_factor
    upscaled_h = resized_h * upscale_factor
    # 16의 배수로 맞춤
    new_w = (
        (upscaled_w // 16) * 16
        if upscaled_w % 16 == 0
        else ((upscaled_w // 16) + 1) * 16
    )
    new_h = (
        (upscaled_h // 16) * 16
        if upscaled_h % 16 == 0
        else ((upscaled_h // 16) + 1) * 16
    )
    control_image = input_image.resize((new_w, new_h), Image.LANCZOS)
    return control_image

def upscale_image(
    input_image,
    prompt,
    negative_prompt,
    upscale_factor,
    guidance_scale,
    num_inference_steps,
    controlnet_conditioning_scale,
    seed,
):
    if input_image is None:
        return None, "이미지를 업로드하세요."

    control_image = resize_image(input_image, upscale_factor)

    # 시드 설정
    if seed != -1:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    try:
        image = upscale_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=control_image.height,
            width=control_image.width,
        ).images[0]

        filename = f"flux1-dev-controlnet-Upscaler08-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)

        info = f"생성 완료!\n저장 파일: {filename}\n최종 크기: {control_image.width}x{control_image.height}\n가이던스 스케일: {guidance_scale}\n추론 스텝: {num_inference_steps}\n컨디셔닝 스케일: {controlnet_conditioning_scale}\n시드: {seed if seed != -1 else '랜덤'}"
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
            prompt_input = gr.Textbox(
                label="프롬프트 (선택)",
                placeholder="이미지에 적용할 스타일이나 설명을 입력하세요...",
                value="8k, high detail, high quality, realistic, masterpiece, best quality",
                lines=2,
            )
            negative_prompt_input = gr.Textbox(
                label="네거티브 프롬프트 (선택)",
                placeholder="피하고 싶은 요소들을 입력하세요 (예: blurry, low quality, artifacts)...",
                value="blurring, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, normal quality, jpeg artifacts, signature, watermark, username",
                lines=2,
            )
            upscale_slider = gr.Radio(
                choices=[1, 2, 4],
                value=2,
                label="업스케일 배율",
                info="이미지를 몇 배로 확대할지 선택",
            )
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=6.5,
                step=0.1,
                label="가이던스 스케일",
                info="프롬프트 준수 정도. 높을수록 프롬프트를 더 정확히 따름.",
            )
            steps_slider = gr.Slider(
                minimum=10,
                maximum=100,
                value=50,
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
            seed_input = gr.Number(
                label="시드 (Seed)",
                value=42,
                precision=0,
                info="랜덤 시드 값. -1이면 랜덤, 고정값을 입력하면 동일한 결과를 재현할 수 있습니다.",
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
            prompt_input,
            negative_prompt_input,
            upscale_slider,
            guidance_slider,
            steps_slider,
            conditioning_slider,
            seed_input,
        ],
        outputs=[output_image, info_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
