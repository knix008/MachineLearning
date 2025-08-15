import torch
import gradio as gr
from diffusers import FluxPipeline
import datetime
import time

# Load model with memory optimizations
print("모델을 로딩 중입니다...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
pipe.enable_vae_slicing()  # Slice VAE computation
print("모델 로딩 완료!")


def generate_image(
    prompt,  # 생성할 이미지에 대한 프롬프트(설명)
    negative_prompt,  # 이미지에서 제외하고 싶은 요소(네거티브 프롬프트)
    width,  # 생성 이미지의 가로 픽셀 크기
    height,  # 생성 이미지의 세로 픽셀 크기
    guidance_scale,  # 프롬프트를 얼마나 강하게 반영할지(높을수록 프롬프트에 더 충실)
    num_inference_steps,  # 이미지 생성 과정의 스텝 수(높을수록 품질↑, 속도↓)
    max_sequence_length,  # 프롬프트 최대 시퀀스 길이(텍스트 길이)
    seed,  # 난수 시드(같은 시드로 같은 결과, -1은 랜덤)
):
    start_time = time.time()

    # 시드 설정
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator("cpu").manual_seed(seed)

    try:
        # 이미지 생성
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,  # 추가
            height=int(height),
            width=int(width),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            max_sequence_length=int(max_sequence_length),
            generator=generator,
        ).images[0]

        end_time = time.time()
        generation_time = end_time - start_time

        # 이미지 저장
        filename = f"Flux1-dev-example04-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(filename)
        info_text = f"생성 완료!\n시간: {generation_time:.2f}초\n시드: {seed}\n저장된 파일: {filename}"

        return image, info_text
    except Exception as e:
        error_text = f"오류 발생: {str(e)}"
        return None, error_text


# Gradio 인터페이스 생성
with gr.Blocks(title="FLUX.1-dev 이미지 생성기") as demo:
    gr.Markdown("# 🎨 FLUX.1-dev 이미지 생성기")
    gr.Markdown("다양한 설정으로 고품질 이미지를 생성해보세요!")

    with gr.Row():
        with gr.Column(scale=1):
            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트 (생성할 이미지를 설명하는 텍스트)",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="a beautiful healthy skinny woman wearing a dark blue bikini, walking on the sunny beach, photo realistic, 8k, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, solo, full body, looking at viewer, long hair, blue eyes, good fingers, good hands, good face, perfect anatomy",
                lines=4,
            )
            negative_prompt_input = gr.Textbox(
                label="네거티브 프롬프트 (이미지에서 제외하고 싶은 요소)",
                placeholder="원하지 않는 요소를 입력하세요...",
                value="low quality, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, jpeg artifacts, signature, watermark, username, blurry",
                lines=2,
            )
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=1024,
                    step=64,
                    label="너비 (생성 이미지의 가로 픽셀 크기)",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=1024,
                    step=64,
                    label="높이 (생성 이미지의 세로 픽셀 크기)",
                )

            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=3.5,
                step=0.1,
                label="가이던스 스케일 (프롬프트 반영 강도, 높을수록 프롬프트에 더 충실)",
            )

            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,
                value=28,
                step=1,
                label="추론 스텝 수 (이미지 생성 단계 수, 높을수록 품질↑, 속도↓)",
            )

            sequence_slider = gr.Slider(
                minimum=128,
                maximum=512,
                value=256,
                step=32,
                label="최대 시퀀스 길이 (프롬프트 최대 텍스트 길이)",
            )
            seed_input = gr.Number(
                label="시드 (-1은 랜덤, 같은 시드로 같은 결과)", value=-1, precision=0
            )

            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            # 출력 영역
            output_image = gr.Image(label="생성된 이미지", type="pil", height=500)
            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    # 이벤트 연결
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            negative_prompt_input,  # 추가
            width_slider,
            height_slider,
            guidance_slider,
            steps_slider,
            sequence_slider,
            seed_input,
        ],
        outputs=[output_image, info_output],
    )

    # 예제 프롬프트
    gr.Examples(
        examples=[
            ["a cute cat holding a sign that says hello world"],
            ["a futuristic city skyline at sunset, cyberpunk style"],
            ["a beautiful landscape with mountains and a lake, oil painting style"],
            ["a portrait of a woman with blue eyes, renaissance painting style"],
            ["a magical forest with glowing mushrooms, fantasy art"],
        ],
        inputs=prompt_input,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
