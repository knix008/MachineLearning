import torch
import gradio as gr
import time
from diffusers import FluxKontextPipeline
from PIL import Image

model_id = "black-forest-labs/FLUX.1-Kontext-dev"

print("모델을 로딩 중입니다...")
pipe = FluxKontextPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# CPU 오프로드 및 Attention 슬라이싱 활성화
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
print("모델 로딩 완료!")


def resize_image(image):
    """
    이미지를 RGB로 변환하고,
    입력 이미지의 가로세로 비율을 유지하면서,
    가로/세로 모두 16의 배수로 맞춤.
    """
    image = image.convert("RGB")
    w, h = image.size
    # 16의 배수로 맞추기 (최대 크기 제한 없음)
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"Resized image to: {new_w}x{new_h}")
    return image


def generate_image(
    prompt, input_image, guidance_scale, steps, seq_len, seed, negative_prompt
):
    """이미지-투-이미지만 지원"""
    start = time.time()

    if input_image is None:
        return None, "이미지-투-이미지만 지원합니다. 입력 이미지를 업로드하세요."

    input_image = resize_image(input_image)  # 이미지 크기 조정

    # 패딩 적용
    info = f"\n입력 이미지 원본 크기: {input_image.size[0]}x{input_image.size[1]}"

    pipe_args = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_image,
        width=input_image.size[0],
        height=input_image.size[1],
        guidance_scale=guidance_scale,
        num_inference_steps=int(steps),
        max_sequence_length=int(seq_len),
        generator=torch.Generator("cpu").manual_seed(
            torch.randint(0, 2**32 - 1, (1,)).item() if seed == -1 else int(seed)
        ),
    )

    try:
        image = pipe(**pipe_args).images[0]
        filename = f"flux1-kontext-dev-example10_{time.strftime('%Y%m%d-%H%M%S')}.png"
        image.save(filename)
        info_text = (
            f"생성 완료! (이미지 투 이미지)\n시간: {time.time()-start:.2f}초\n시드: {seed}\n저장된 파일: {filename}"
            f"\n생성된 이미지 크기: {image.size[0]}x{image.size[1]}{info}"
        )
        return image, info_text
    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# Gradio 인터페이스 생성
with gr.Blocks(title="FLUX.1 Kontext Dev 이미지 생성기") as demo:
    gr.Markdown("# 🎨 FLUX.1-dev 이미지 생성기")

    with gr.Row():
        with gr.Column():
            # 입력 이미지 (선택사항)
            input_image = gr.Image(
                label="입력 이미지 (선택사항)",
                type="pil",
                sources=["upload", "clipboard"],
                height=500,
                value="default.jpg",
            )

            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="8k, animation style, high detail, high quality, detail skin, photo realistic, masterpiece, best quality, dark blue bikini, intricate details",
                lines=4,
            )

            negative_prompt_input = gr.Textbox(
                label="네거티브 프롬프트",
                placeholder="원하지 않는 요소를 입력하세요...",
                value="blurring, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, normal quality, jpeg artifacts, signature, watermark, username",
                lines=2,
            )

            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=3.5,
                step=0.1,
                label="가이던스 스케일",
                info="프롬프트 준수 정도. 높을수록 프롬프트를 더 정확히 따르지만 창의성이 줄어들 수 있습니다. (권장: 3.5-7.0)",
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
                value=100,
                precision=0,
                info="생성 결과의 일관성을 위한 난수 시드. 같은 시드로 같은 설정이면 비슷한 결과가 나옵니다. -1은 무작위 시드 사용",
            )

            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

        with gr.Column():
            # 출력 영역
            output_image = gr.Image(label="생성된 이미지", type="pil", height=500)

            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    # 이벤트 연결
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            input_image,
            guidance_slider,
            steps_slider,
            sequence_slider,
            seed_input,
            negative_prompt_input,  # 추가
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
    demo.launch(inbrowser=True)
