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

MAX_IMAGE_SIZE = 512  # 최대 이미지 크기

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

    # 입력 이미지 크기
    w, h = input_image.size
    w = (w // 8) * 8  # 8의 배수로 조정
    h = (h // 8) * 8  # 8의 배수로 조정

    # 최대 크기 512로 리사이즈 (비율 유지)
    max_dim = max(w, h)
    if max_dim > 512:
        scale = 512 / max_dim
        w = int(w * scale)
        h = int(h * scale)
        input_image = input_image.resize((w, h), Image.LANCZOS)

    # 업스케일: 비율 유지 (w/h 비율 그대로)
    new_w = int(w * upscale_factor)
    new_h = int(h * upscale_factor)
    resized_image = input_image.resize((new_w, new_h), Image.LANCZOS)

    print(f"조정된 이미지 크기: {w}x{h}, 최종 크기: {new_w}x{new_h}")

    # 시드 설정
    if seed != -1:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    try:
        image = upscale_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            control_image=resized_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=new_h,
            width=new_w,
        ).images[0]

        filename = f"flux1-dev-controlnet-Upscaler05-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)

        info = f"생성 완료!\n저장 파일: {filename}\n조정된 이미지 크기: {w}x{h}\n최종 크기: {new_w}x{new_h}\n가이던스 스케일: {guidance_scale}\n추론 스텝: {num_inference_steps}\n컨디셔닝 스케일: {controlnet_conditioning_scale}\n시드: {seed if seed != -1 else '랜덤'}"
        
        

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
                value="8k, high detail, high quality, photo realistic, masterpiece, best quality",
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
                value=3.5,
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
            output_image = gr.Image(label="업스케일 결과", type="pil", height=500)
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
