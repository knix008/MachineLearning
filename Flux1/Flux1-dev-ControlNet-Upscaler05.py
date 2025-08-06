import torch
from diffusers import FluxControlNetModel
from diffusers.pipelines import FluxControlNetPipeline
import datetime
import gradio as gr
from PIL import Image

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
print("모델 로딩 완료!")

MAX_IMAGE_SIZE = 512  # 최대 이미지 크기

def resize_image(input_image, upscale_factor):
        # 입력 이미지 크기
    w, h = input_image.size

    # 최대 크기 512로 리사이즈 (비율 유지)
    max_dim = max(w, h)
    if max_dim > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max_dim
        w = int(w * scale)
        h = int(h * scale)
        input_image = input_image.resize((w, h), Image.LANCZOS)

    # 업스케일: 비율 유지 (w/h 비율 그대로)
    new_w = int(w * upscale_factor)
    new_h = int(h * upscale_factor)
    resized_image = input_image.resize((new_w, new_h), Image.LANCZOS)
    return resized_image, w, h, new_w, new_h

def upscale_image(
    input_image,
    upscale_factor,
    guidance_scale,
    num_inference_steps,
    controlnet_conditioning_scale,
    prompt,
):
    if input_image is None:
        return None, "이미지를 업로드하세요."

    resized_image, w, h, new_w, new_h = resize_image(input_image, upscale_factor)

    try:
        image = pipe(
            prompt=prompt,
            control_image=resized_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=new_h,
            width=new_w,
        ).images[0]

        filename = f"Flux1-Upscaled-Example05-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)

        info = f"생성 완료!\n저장 파일: {filename}\n입력 크기: {w}x{h}\n최종 크기: {new_w}x{new_h}\n가이던스 스케일: {guidance_scale}\n추론 스텝: {num_inference_steps}\n컨디셔닝 스케일: {controlnet_conditioning_scale}"
        return image, info
    except Exception as e:
        return None, f"오류 발생: {str(e)}"

with gr.Blocks(title="FLUX.1 ControlNet 업스케일러") as demo:
    gr.Markdown("# 🖼️ FLUX.1 ControlNet 업스케일러")

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
                value="8k, high detail, realistic, high quality, masterpiece, best quality",
                lines=2,
            )
            upscale_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                label="업스케일 배율",
                info="이미지를 몇 배로 확대할지 선택 (예: 4배)",
            )
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=6.0,
                step=0.1,
                label="가이던스 스케일",
                info="프롬프트 준수 정도. 높을수록 프롬프트를 더 정확히 따름.",
            )
            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,
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
            generate_btn = gr.Button("🖼️ 업스케일 이미지 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="업스케일 결과", type="pil", height=512)
            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    generate_btn.click(
        fn=upscale_image,
        inputs=[
            input_image,
            upscale_slider,
            guidance_slider,
            steps_slider,
            conditioning_slider,
            prompt_input,
        ],
        outputs=[output_image, info_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
