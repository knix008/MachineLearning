import torch
import gradio as gr
import time
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
import os

# Dependency!!! :
# You need to install the diffusers with the following command:
# pip install git+https://github.com/huggingface/diffusers.git

# Load model with memory optimizations
print("모델을 로딩 중입니다...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
pipe.enable_vae_slicing()  # Slice VAE computation
print("모델 로딩 완료!")


# 기본 이미지 로드 함수
def load_default_image():
    """기본 이미지 파일이 존재하면 로드"""
    default_path = "default.jpg"
    if os.path.exists(default_path):
        try:
            return load_image(default_path)
        except Exception as e:
            print(f"기본 이미지 로드 실패: {e}")
            return None
    return None


def generate_image(
    prompt,
    input_image,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    max_sequence_length,
    strength,
    seed,
):
    """이미지 생성 함수 (텍스트-투-이미지 또는 이미지-투-이미지)"""
    start_time = time.time()

    # 입력 이미지가 있는 경우 해당 이미지의 비율 사용
    if input_image is not None:
        input_width, input_height = input_image.size
        aspect_ratio = input_width / input_height

        # 입력 이미지 비율에 맞춰 크기 조정
        if aspect_ratio >= 1.0:  # 가로가 더 크거나 같은 경우
            adjusted_width = max(768, int(width))
            adjusted_height = int(adjusted_width / aspect_ratio)
        else:  # 세로가 더 큰 경우
            adjusted_height = max(768, int(height))
            adjusted_width = int(adjusted_height * aspect_ratio)

        # 16의 배수로 조정
        adjusted_width = (adjusted_width // 16) * 16
        adjusted_height = (adjusted_height // 16) * 16

        # 최소 크기 보장
        adjusted_width = max(adjusted_width, 512)
        adjusted_height = max(adjusted_height, 512)

        generation_type = "이미지 투 이미지"

    else:
        # 기본 이미지가 있는지 확인
        default_image = load_default_image()
        if default_image is not None:
            # 기본 이미지의 비율 사용
            default_width, default_height = default_image.size
            aspect_ratio = default_width / default_height

            # 기본 이미지 비율에 맞춰 크기 조정
            if aspect_ratio >= 1.0:  # 가로가 더 크거나 같은 경우
                adjusted_width = max(768, int(width))
                adjusted_height = int(adjusted_width / aspect_ratio)
            else:  # 세로가 더 큰 경우
                adjusted_height = max(768, int(height))
                adjusted_width = int(adjusted_height * aspect_ratio)

            # 16의 배수로 조정
            adjusted_width = (adjusted_width // 16) * 16
            adjusted_height = (adjusted_height // 16) * 16

            # 최소 크기 보장
            adjusted_width = max(adjusted_width, 512)
            adjusted_height = max(adjusted_height, 512)

            generation_type = "기본이미지-투-이미지"
        else:
            # 사용자 지정 크기 사용
            width = int(width)
            height = int(height)
            aspect_ratio = width / height

            # 16의 배수로 조정
            adjusted_width = (width // 16) * 16
            adjusted_height = (height // 16) * 16

            # 최소 크기 보장하면서 비율 유지
            if adjusted_width < 512 or adjusted_height < 512:
                if aspect_ratio >= 1.0:  # 가로가 더 크거나 같은 경우
                    adjusted_height = 512
                    adjusted_width = int(512 * aspect_ratio)
                    adjusted_width = (adjusted_width // 16) * 16
                else:  # 세로가 더 큰 경우
                    adjusted_width = 512
                    adjusted_height = int(512 / aspect_ratio)
                    adjusted_height = (adjusted_height // 16) * 16

            # 최종 최소 크기 확인
            adjusted_width = max(adjusted_width, 512)
            adjusted_height = max(adjusted_height, 512)

            generation_type = "텍스트 투 이미지"

    # 시드 설정
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator("cpu").manual_seed(seed)

    try:
        # 입력 이미지가 있는 경우 img2img, 없는 경우 txt2img
        if input_image is not None:
            # 입력 이미지 크기 조정
            input_image = input_image.resize(
                (adjusted_width, adjusted_height), Image.LANCZOS
            )

            # img2img 생성
            image = pipe(
                prompt=prompt,
                image=input_image,
                height=adjusted_height,
                width=adjusted_width,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                max_sequence_length=int(max_sequence_length),
                generator=generator,
            ).images[0]

        else:
            # 기본 이미지가 있는지 확인
            default_image = load_default_image()
            if default_image is not None:
                # 기본 이미지를 사용한 img2img
                default_image = default_image.resize(
                    (adjusted_width, adjusted_height), Image.LANCZOS
                )

                image = pipe(
                    prompt=prompt,
                    image=default_image,
                    height=adjusted_height,
                    width=adjusted_width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=int(num_inference_steps),
                    max_sequence_length=int(max_sequence_length),
                    generator=generator,
                ).images[0]

            else:
                # txt2img 생성
                image = pipe(
                    prompt=prompt,
                    height=adjusted_height,
                    width=adjusted_width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=int(num_inference_steps),
                    max_sequence_length=int(max_sequence_length),
                    generator=generator,
                ).images[0]

        end_time = time.time()
        generation_time = end_time - start_time

        # 이미지 저장
        timestamp = int(time.time())
        filename = f"flux_generated_{timestamp}.png"
        image.save(filename)

        # 생성된 이미지 크기 정보
        generated_width, generated_height = image.size

        # 크기 조정 정보 포함
        size_info = f"\n생성된 이미지 크기: {generated_width}x{generated_height}"
        if input_image is not None:
            original_width, original_height = input_image.size
            size_info += f"\n입력 이미지 크기: {original_width}x{original_height}"
            size_info += f"\n비율 맞춤: {original_width/original_height:.2f} → {generated_width/generated_height:.2f}"
        elif default_image is not None:
            default_width, default_height = default_image.size
            size_info += f"\n기본 이미지 크기: {default_width}x{default_height}"
            size_info += f"\n비율 맞춤: {default_width/default_height:.2f} → {generated_width/generated_height:.2f}"
        else:
            size_info += f"\n요청 크기: {width}x{height}"

        info_text = f"생성 완료! ({generation_type})\n시간: {generation_time:.2f}초\n시드: {seed}\n저장된 파일: {filename}{size_info}"

        return image, info_text

    except Exception as e:
        error_text = f"오류 발생: {str(e)}"
        return None, error_text


def update_ui_visibility(input_image):
    """입력 이미지에 따라 UI 요소 표시/숨김 및 기본 이미지 표시"""
    if input_image is not None:
        return gr.update(visible=False), gr.update(
            value="이미지를 프롬프트에 맞게 수정합니다..."
        )
    else:
        # 기본 이미지가 있는지 확인하고 표시
        default_image = load_default_image()
        if default_image is not None:
            return gr.update(visible=False), gr.update(
                value="기본 이미지를 사용하여 프롬프트에 맞게 수정합니다..."
            )
        else:
            return gr.update(visible=False), gr.update(
                value="생성하고 싶은 이미지를 설명해주세요..."
            )


# Gradio 인터페이스 생성
with gr.Blocks(title="FLUX.1-dev 이미지 생성기") as demo:
    gr.Markdown("# 🎨 FLUX.1-dev 이미지 생성기")
    gr.Markdown(
        "텍스트로 새 이미지를 생성하거나, 기존 이미지를 프롬프트에 맞게 수정하세요!"
    )

    # 기본 이미지 로드
    default_img = load_default_image()

    with gr.Row():
        with gr.Column(scale=1):
            default_prompt = "photorealistic, 8k resolution, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, looking at viewer, perfect anatomy"
            # 입력 이미지 (선택사항)
            input_image = gr.Image(
                label="입력 이미지 (선택사항)",
                type="pil",
                sources=["upload", "clipboard"],
                value=default_img,
            )

            # 기본 이미지 상태 표시
            if default_img is not None:
                gr.Markdown(
                    "💡 **기본 이미지가 로드되었습니다.** 이미지를 업로드하지 않으면 기본 이미지를 사용합니다."
                )

            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="skinny, blue eyes, full body, beautiful face, good body shape, good hair, good fingers, good legs",
                lines=4,
            )
            
            prompt = prompt_input + default_prompt

            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256, maximum=1024, value=768, step=64, label="너비"
                )
                height_slider = gr.Slider(
                    minimum=256, maximum=1024, value=768, step=64, label="높이"
                )

            guidance_slider = gr.Slider(
                minimum=1.0, maximum=10.0, value=3.5, step=0.1, label="가이던스 스케일"
            )

            steps_slider = gr.Slider(
                minimum=10, maximum=50, value=28, step=1, label="추론 스텝 수"
            )

            sequence_slider = gr.Slider(
                minimum=128, maximum=512, value=256, step=32, label="최대 시퀀스 길이"
            )

            # 이미지-투-이미지 전용 설정
            strength_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,  # 기본값을 0.5로 낮춤 (원본 더 보존)
                step=0.1,
                label="변형 강도 (낮을수록 원본 유지)",
                visible=True if default_img is not None else False,
            )

            seed_input = gr.Number(label="시드 (-1은 랜덤)", value=-1, precision=0)
            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            # 출력 영역
            output_image = gr.Image(label="생성된 이미지", type="pil", height=500)
            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    # 이벤트 연결
    input_image.change(
        fn=update_ui_visibility,
        inputs=[input_image],
        outputs=[strength_slider, prompt_input],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            input_image,
            width_slider,
            height_slider,
            guidance_slider,
            steps_slider,
            sequence_slider,
            strength_slider,
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
            ["convert this image to anime style, vibrant colors"],
            ["make this image look like a watercolor painting"],
            ["transform this to a cyberpunk style with neon lights"],
        ],
        inputs=prompt_input,
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
