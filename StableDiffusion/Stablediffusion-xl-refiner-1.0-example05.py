import torch
import gradio as gr
from diffusers import StableDiffusionXLImg2ImgPipeline
import datetime

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    )
refiner.enable_model_cpu_offload()
refiner.enable_sequential_cpu_offload()
refiner.enable_attention_slicing()

print("> stable-diffusion-xl-refiner-1.0 모델 로드 성공")


def generate_image(input_image, prompt, negative_prompt, strength, guidance_scale, num_inference_steps):
    try:
        # 2단계: 리파이너로 이미지 정제
        refined_image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            width=input_image.width,
            height=input_image.height,
            num_inference_steps=num_inference_steps,
            denoising_start=0.8,  # 베이스에서 80% 완료된 지점부터 시작
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        # 이미지 저장 (선택 사항)
        refined_image.save(f"sdxl-refiner-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
        return refined_image
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


# Gradio 인터페이스 생성
with gr.Blocks(title="Stable Diffusion XL Refiner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Stable Diffusion XL Refiner")
    gr.Markdown("프롬프트를 입력하여 리파이너로 고품질 이미지를 만들어보세요.")

    with gr.Row():
        with gr.Column():
            # 입력 컨트롤
            input_image = gr.Image(
                label="입력 이미지 (Input Image)",
                type="pil",
                height=500,
                value="default.jpg",  # 기본 이미지 파일 경로
            )

            prompt = gr.Textbox(
                label="Prompt (프롬프트)",
                lines=3,
                value="8k uhd, high detail, high quality, ultra high resolution",
                info="이미지에 원하는 특징, 스타일, 분위기 등을 영어로 입력하세요."
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt (네거티브 프롬프트)",
                placeholder="예: blurry, low quality, distorted, ugly",
                lines=3,
                value="blurry, low quality, distorted, ugly, deformed, bad anatomy",
                info="이미지에서 피하고 싶은 요소를 영어로 입력하세요."
            )


            strength = gr.Slider(
                label="Refiner Strength (리파이너 강도)",
                minimum=0.1,
                maximum=1.0,
                value=0.80,
                step=0.05,
                info="리파이너가 이미지를 얼마나 많이 수정할지 결정합니다. 낮을수록 원본 유지, 높을수록 변화 큼."
            )

            guidance_scale = gr.Slider(
                label="Guidance Scale (프롬프트 반영 정도)",
                minimum=1.0,
                maximum=20.0,
                value=7.5,
                step=0.1,
                info="프롬프트를 얼마나 강하게 반영할지 결정합니다. 너무 높으면 부자연스러울 수 있습니다."
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps (생성 단계 수)",
                minimum=20,
                maximum=50,
                value=50,
                step=1,
                info="이미지 생성 품질과 속도에 영향을 줍니다. 높을수록 품질이 좋아지지만 시간이 오래 걸립니다."
            )

            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

        with gr.Column():
            refined_output = gr.Image(
                label="리파인된 이미지 (Refined)",
                height=500,
            )

    generate_btn.click(
        fn=generate_image,
        inputs=[input_image, prompt, negative_prompt, strength, guidance_scale, num_inference_steps],
        outputs=[refined_output],
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)