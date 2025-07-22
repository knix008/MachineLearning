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


def generate_image(
    prompt,
    input_image,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    max_sequence_length,
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
        # 사용자 지정 크기 사용 (텍스트-투-이미지)
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
            # 원본 input_image 정보 사용 (리사이즈 전)
            size_info += (
                f"\n입력 이미지 크기: {input_image.size[0]}x{input_image.size[1]}"
            )
            size_info += f"\n비율 맞춤: {input_image.size[0]/input_image.size[1]:.2f} → {generated_width/generated_height:.2f}"
        else:
            size_info += f"\n요청 크기: {width}x{height}"

        info_text = f"생성 완료! ({generation_type})\n시간: {generation_time:.2f}초\n시드: {seed}\n저장된 파일: {filename}{size_info}"

        return image, info_text

    except Exception as e:
        error_text = f"오류 발생: {str(e)}"
        return None, error_text


# Gradio 인터페이스 생성
with gr.Blocks(title="FLUX.1-dev 이미지 생성기") as demo:
    gr.Markdown("# 🎨 FLUX.1-dev 이미지 생성기")
    gr.Markdown(
        "텍스트로 새 이미지를 생성하거나, 기존 이미지를 프롬프트에 맞게 수정하세요!"
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 입력 이미지 (선택사항)
            input_image = gr.Image(
                label="입력 이미지 (선택사항)",
                type="pil",
                sources=["upload", "clipboard"],
            )

            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="blue bikini, blue eyes, beautiful face, good body shape, good hair, good fingers, good legs, photorealistic, 8k resolution, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, looking at viewer, perfect anatomy",
                lines=4,
            )

            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64,
                    label="너비",
                    info="생성할 이미지의 너비 (픽셀). 높을수록 더 넓은 이미지가 생성됩니다.",
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64,
                    label="높이",
                    info="생성할 이미지의 높이 (픽셀). 높을수록 더 긴 이미지가 생성됩니다.",
                )

            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=3.5,
                step=0.1,
                label="가이던스 스케일",
                info="프롬프트 준수 정도. 높을수록 프롬프트를 더 정확히 따르지만 창의성이 줄어들 수 있습니다. (권장: 3.5-7.5)",
            )

            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,
                value=28,
                step=1,
                label="추론 스텝 수",
                info="이미지 생성 단계 수. 높을수록 품질이 향상되지만 생성 시간이 늘어납니다. (권장: 20-30)",
            )

            sequence_slider = gr.Slider(
                minimum=128,
                maximum=512,
                value=256,
                step=32,
                label="최대 시퀀스 길이",
                info="프롬프트 처리 길이. 긴 프롬프트에는 높은 값이 필요합니다. (기본: 256)",
            )

            seed_input = gr.Number(
                label="시드 (-1은 랜덤)",
                value=-1,
                precision=0,
                info="생성 결과의 일관성을 위한 난수 시드. 같은 시드로 같은 설정이면 비슷한 결과가 나옵니다. -1은 무작위 시드 사용",
            )

            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

            # 설정 가이드 추가
            with gr.Accordion("📚 설정 가이드", open=False):
                gr.Markdown(
                    """
                ### 🎯 주요 설정 설명
                
                **🎨 가이던스 스케일 (Guidance Scale)**
                - 1.0-3.0: 창의적이고 다양한 결과, 프롬프트를 느슨하게 따름
                - 3.5-7.5: 균형잡힌 결과 (권장)
                - 8.0-10.0: 프롬프트를 매우 정확히 따르지만 창의성 부족
                
                **⚡ 추론 스텝 수 (Inference Steps)**
                - 10-15: 빠른 생성, 낮은 품질
                - 20-30: 균형잡힌 품질과 속도 (권장)
                - 35-50: 높은 품질, 긴 생성 시간
                
                **📏 최대 시퀀스 길이 (Max Sequence Length)**
                - 128-192: 짧은 프롬프트용, 빠른 처리
                - 256: 표준 길이, 대부분의 프롬프트에 적합 (권장)
                - 320-512: 긴 프롬프트용, 복잡한 설명 처리 가능
                - 높을수록 더 긴 프롬프트를 정확히 처리하지만 처리 시간 증가
                
                **🎲 시드 (Seed)**
                - -1: 매번 다른 결과 생성
                - 고정값: 같은 설정으로 일관된 결과 생성
                - 좋은 결과가 나오면 시드를 기록해두세요!
                """
                )

        with gr.Column(scale=1):
            # 출력 영역
            output_image = gr.Image(label="생성된 이미지", type="pil", height=500)

            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    # 이벤트 연결
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

    # 사용 팁 추가
    with gr.Accordion("💡 사용 팁", open=False):
        gr.Markdown(
            """
        ### 🚀 효과적인 사용법
        
        **📝 프롬프트 작성 팁**
        - 구체적이고 명확한 설명 사용
        - 스타일 키워드 포함: "photorealistic", "oil painting", "anime style" 등
        - 품질 키워드 추가: "high quality", "detailed", "masterpiece" 등
        - 긴 프롬프트 사용 시 최대 시퀀스 길이를 320-512로 증가
        
        **🎨 새 이미지 생성 (Text-to-Image)**
        - 가이던스 스케일: 3.5-7.5
        - 추론 스텝: 25-30
        - 해상도: 768x768 또는 1024x1024
        - 최대 시퀀스 길이: 256 (표준), 긴 프롬프트 시 512
        
        **🖼️ 이미지 수정 (Image-to-Image)**
        - 업로드한 이미지를 기반으로 프롬프트에 맞게 수정
        - 원본 이미지의 구조와 내용을 유지하면서 변형
        - 복잡한 수정 요청 시 시퀀스 길이를 높여보세요
        
        **⚡ 성능 최적화**
        - 메모리 부족 시: 해상도를 512x512로 낮추기
        - 빠른 생성: 추론 스텝 15-20, 시퀀스 길이 192-256
        - 고품질 생성: 추론 스텝 30-40, 시퀀스 길이 256-320
        """
        )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
