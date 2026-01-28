import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gradio as gr

DEFAULT_PROMPT = "a glamorous red bikini swimsuit hot skinny korean model posing on a tropical beach at sunset, cinematic lighting, 4k, ultra-detailed texture, with perfect anatomy, fashion vibe."
DEFAULT_IMAGE = "default.png"

# Global variables for model
DEVICE = "cuda"
DTYPE = torch.bfloat16
pipe = None

def load_model():
    """Load and initialize the Flux2 Img2Img model with optimizations."""
    global pipe

    print("모델 로딩 중...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        torch_dtype=DTYPE
    )
    pipe = pipe.to(DEVICE)

    # Memory optimization
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload() # 안쓰면 CUDA에서 더 빠름(4 추론스텝에서 1초 단축), CPU에서는 사용해야 함

    print("모델 로딩 완료!")
    return pipe

def generate_image(input_image, prompt, strength, guidance_scale, num_inference_steps, seed):
    """Generate image from input image and text prompt."""
    global pipe

    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        # Convert to PIL Image if needed
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)

        # Setup generator
        generator = torch.Generator(device=DEVICE)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        print(f"이미지 편집 중... (steps: {num_inference_steps}, strength: {strength})")

        # Generate image
        image = pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}_seed{int(seed)}.png"
        image.save(output_path)
        print(f"이미지 저장됨: {output_path}")

        return image, f"✓ 이미지 편집 완료! 저장됨: {output_path}"

    except Exception as e:
        return None, f"오류: {str(e)}"

def main():
    # Load model once at startup
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.2 Dev Image-to-Image 편집기") as demo:
        gr.Markdown("# Flux.2 Dev Image-to-Image 편집기")
        gr.Markdown("이미지를 업로드하고 프롬프트로 편집하세요.")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="입력 이미지",
                    type="pil",
                    value=DEFAULT_IMAGE,
                    height=800
                )
                prompt_input = gr.Textbox(
                    label="프롬프트 (영어)",
                    info="편집할 내용을 설명하세요",
                    placeholder="예: change the background to a beach at sunset",
                    value=DEFAULT_PROMPT,
                    lines=3
                )

                with gr.Accordion("고급 설정", open=True):
                    strength_input = gr.Slider(
                        label="Strength (강도)",
                        info="원본 이미지 변형 정도 (0: 원본 유지, 1: 완전히 새로 생성)",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.75
                    )

                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=1.0,
                            maximum=20.0,
                            step=0.5,
                            value=7.5
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=10,
                            maximum=20,
                            step=1,
                            value=4
                        )

                    seed_input = gr.Slider(
                        label="시드 (Seed)",
                        info="재현성을 위한 난수 시드 (-1: 랜덤)",
                        minimum=-1,
                        maximum=1000,
                        step=1,
                        value=42
                    )

                submit_btn = gr.Button("이미지 편집", variant="primary", size="lg")

            with gr.Column():
                image_output = gr.Image(label="출력 이미지", height=800)
                status_output = gr.Textbox(label="상태", interactive=False)

        # Connect button to generation function
        submit_btn.click(
            fn=generate_image,
            inputs=[
                input_image,
                prompt_input,
                strength_input,
                guidance_input,
                steps_input,
                seed_input
            ],
            outputs=[image_output, status_output]
        )

    # Launch the interface
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()
