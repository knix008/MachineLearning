import torch
from diffusers import FluxKontextPipeline
from datetime import datetime
import os
import atexit
import gradio as gr

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set device and data type
device = "mps"
dtype = torch.bfloat16

print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=dtype
)
pipe.to(device)
print("Model loaded!")


def cleanup():
    """Release all resources."""
    global pipe
    print("\nCleaning up resources...")
    try:
        del pipe
    except NameError:
        pass
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Resources released!")


# Register cleanup function to run on exit
atexit.register(cleanup)

default_prompt = "Add a beach background with palm trees and a bright sunny sky."


def generate_image(
    input_image, prompt, width, height, guidance_scale, num_inference_steps, seed, strength, max_sequence_length
):
    try:
        if input_image is None:
            return None, "✗ 입력 이미지를 업로드해주세요."

        # Create generator with seed
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Run the pipeline
        image = pipe(
            image=input_image,
            prompt=prompt,
            width=int(width),
            height=int(height),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            strength=strength,
            generator=generator,
            max_sequence_length=int(max_sequence_length),
        ).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = os.path.join(script_dir, f"{script_name}_{timestamp}.png")
        image.save(filename)

        return image, f"✓ 이미지가 저장되었습니다: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Flux.1-Kontext Image-to-Image Generator") as interface:
    gr.Markdown("# 🎨 Flux.1-Kontext Image-to-Image Generator")
    gr.Markdown("입력 이미지를 기반으로 AI를 사용하여 새로운 이미지를 생성합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            # Input image
            input_image = gr.Image(
                label="입력 이미지",
                type="pil",
                height=300,
            )

            # Prompt
            prompt = gr.Textbox(
                label="프롬프트",
                value=default_prompt,
                lines=3,
                placeholder="이미지에 적용할 변경 사항을 입력하세요",
                info="입력 이미지에 적용하고 싶은 변경 사항을 설명합니다.",
            )

            with gr.Row():
                width = gr.Slider(
                    label="이미지 너비",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=768,
                    info="생성할 이미지의 너비 (픽셀). 64의 배수.",
                )
                height = gr.Slider(
                    label="이미지 높이",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024,
                    info="생성할 이미지의 높이 (픽셀). 64의 배수.",
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=2.5,
                    info="프롬프트를 얼마나 따를지 제어. 낮을수록 창의적, 높을수록 정확.",
                )
                num_inference_steps = gr.Slider(
                    label="추론 스텝",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=28,
                    info="생성 단계 수. 높을수록 품질 향상, 시간 증가.",
                )

            with gr.Row():
                seed = gr.Number(
                    label="시드",
                    value=42,
                    precision=0,
                    info="같은 시드 = 같은 결과. 재현성을 위해 사용.",
                )
                strength = gr.Slider(
                    label="강도 (Strength)",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.95,
                    info="원본 이미지 변형 정도. 높을수록 더 많이 변형.",
                )

            with gr.Row():
                max_sequence_length = gr.Slider(
                    label="Max Sequence Length",
                    minimum=128,
                    maximum=512,
                    step=64,
                    value=512,
                    info="텍스트 인코더 시퀀스 길이. 긴 프롬프트는 높은 값 필요.",
                )

            generate_btn = gr.Button("🚀 이미지 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(label="생성된 이미지", height=500)
            output_message = gr.Textbox(label="상태", interactive=False)

    # Connect the generate button to the function
    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image,
            prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            seed,
            strength,
            max_sequence_length,
        ],
        outputs=[output_image, output_message],
    )

    gr.Markdown("---")
    gr.Markdown(
        """
### 파라미터 설명:

**입력 이미지** - 변형할 원본 이미지를 업로드합니다.

**프롬프트** - 이미지에 적용할 변경 사항을 설명합니다. 예: "배경을 해변으로 변경"

**이미지 크기 (Width/Height)** - 출력 이미지의 너비와 높이 (256-1024px, 64의 배수)

**Guidance Scale** - 프롬프트 따르기 강도. 낮을수록 창의적, 높을수록 정확. 권장: 2-5

**추론 스텝** - 생성 단계 수. 높을수록 품질 향상, 시간 증가. 권장: 20-28

**시드** - 난수 시드. 같은 시드로 같은 결과 재현 가능.

**강도 (Strength)** - 원본 이미지 변형 정도. 0.1=거의 유지, 1.0=완전히 변형. 권장: 0.7-0.95

**Max Sequence Length** - 텍스트 인코더의 최대 시퀀스 길이. 긴 프롬프트는 높은 값 필요. 권장: 256-512
    """
    )

# Launch the interface
if __name__ == "__main__":
    try:
        interface.launch(inbrowser=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        cleanup()
