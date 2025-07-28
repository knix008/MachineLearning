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

def resize_image_keep_aspect(image, max_size=512):
    """이미지 비율을 유지하면서 최대 크기 제한"""
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    
    # 비율 계산
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    
    return image.resize((new_w, new_h), Image.LANCZOS)

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
    
    # 입력 이미지를 512 이하로 리사이즈 (비율 유지)
    resized_input = resize_image_keep_aspect(input_image, max_size=512)
    w, h = resized_input.size
    
    # 업스케일 적용
    new_w = int(w * upscale_factor)
    new_h = int(h * upscale_factor)
    resized_image = resized_input.resize((new_w, new_h), Image.LANCZOS)

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
        filename = f"Flux1-Upscaled-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)
        info = f"생성 완료!\n저장 파일: {filename}\n입력 크기: {input_image.size[0]}x{input_image.size[1]}\n리사이즈 후: {w}x{h}\n최종 크기: {new_w}x{new_h}\n가이던스 스케일: {guidance_scale}\n추론 스텝: {num_inference_steps}\n컨디셔닝 스케일: {controlnet_conditioning_scale}"
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
                height=400,
                value="default.jpg",  # 기본 이미지 경로 (예시용)
            )
            prompt_input = gr.Textbox(
                label="프롬프트 (선택)",
                placeholder="이미지에 적용할 스타일이나 설명을 입력하세요...",
                value="8k, high detail, realistic, high quality, masterpiece, best quality, detailed, intricate, smooth, cinematic lighting",
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
            generate_btn = gr.Button("🖼️ 업스케일 이미지 생성", variant="primary", size="lg")

            with gr.Accordion("📚 설정 가이드", open=False):
                gr.Markdown(
                    """
                    ### 주요 설정 설명

                    **업스케일 배율**  
                    - 이미지를 몇 배로 확대할지 선택합니다.  
                    - 너무 크게 하면 메모리 부족이 발생할 수 있습니다.

                    **가이던스 스케일**  
                    - 프롬프트를 얼마나 엄격하게 따를지 결정합니다.  
                    - 높을수록 프롬프트에 더 가까운 결과가 나오지만 창의성이 줄어듭니다.

                    **추론 스텝 수**  
                    - 이미지 생성 단계 수입니다.  
                    - 높을수록 품질이 좋아지지만 시간이 오래 걸립니다.

                    **컨디셔닝 스케일**  
                    - ControlNet의 영향력입니다.  
                    - 높을수록 입력 이미지의 구조를 더 강하게 반영합니다.

                    **프롬프트**  
                    - 이미지에 적용할 스타일이나 설명을 입력하세요.  
                    - 예시: "anime style, vibrant colors, high detail"
                    """
                )

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
