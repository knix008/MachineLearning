import torch
import gradio as gr
import time
from diffusers import FluxKontextPipeline
from PIL import Image
import datetime


def loading_model():
    model_id = "black-forest-labs/FLUX.1-Kontext-dev"
    print("모델을 로딩 중입니다...")
    pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    # CPU 오프로드 및 Attention 슬라이싱 활성화
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    print("모델 로딩 완료!")
    return pipe


# 로딩된 모델을 전역 변수로 저장
pipe = loading_model()

MAX_IMAGE_SIZE = 1024
def resize_image(input_image):
    w, h = input_image.size
    max_side = max(w, h)
    if max_side > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max_side
        w_new = int(w * scale)
        h_new = int(h * scale)
    else:
        w_new, h_new = w, h
    # Make both dimensions multiples of 16
    w_new = (w_new // 16) * 16
    h_new = (h_new // 16) * 16
    # Avoid zero size
    w_new = max(w_new, 16)
    h_new = max(h_new, 16)
    resized_image = input_image.resize((w_new, h_new), Image.Resampling.LANCZOS)
    return resized_image


def generate_image(
    prompt,
    negative_prompt,
    input_image,
    guidance_scale,
    steps,
    seq_len,
    seed,
):
    start = time.time()

    if input_image is None:
        return None, "이미지-투-이미지만 지원합니다. 입력 이미지를 업로드하세요."

    input_image = resize_image(input_image)  # 이미지 크기 조정
    info = f"\n입력 이미지 크기: {input_image.size[0]}x{input_image.size[1]}"
    # Generator 설정
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator("cpu").manual_seed(seed)

    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            width=input_image.size[0],
            height=input_image.size[1],
            guidance_scale=guidance_scale,
            num_inference_steps=int(steps),
            max_sequence_length=int(seq_len),
            generator=generator,
        ).images[0]

        # 고품질 저장 설정
        filename = f"Flux1-Kontext-dev-example15_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
        image.save(
            filename, format="PNG", compress_level=1, optimize=False
        )  # 최고 품질로 저장
        info_text = (
            f"생성 완료! (이미지 투 이미지)\n시간: {time.time()-start:.2f}초\n시드: {seed}\n저장된 파일: {filename}"
            f"\n생성된 이미지 크기: {image.size[0]}x{image.size[1]}{info}"
        )
        return image, info_text
    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# Gradio 인터페이스 생성
with gr.Blocks(title="FLUX.1 Kontext Dev 이미지 생성기") as demo:
    gr.Markdown("# 🎨 FLUX.1 Kontext Dev이미지 생성기")

    with gr.Row():
        with gr.Column():
            # 입력 이미지 (선택사항)
            input_image = gr.Image(
                label="입력 이미지 (선택사항)",
                type="pil",
                sources=["upload", "clipboard"],
                height=500,
                value="default16.jpg",
            )

            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="8k, high detail, high quality, best quality, masterpiece, dark blue bikini",
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
                maximum=10.0,  # 범위 확장
                value=3.5,  # 더 높은 기본값
                step=0.1,
                label="가이던스 스케일",
                info="프롬프트 준수 정도. 높을수록 프롬프트를 더 정확히 따르지만 창의성이 줄어들 수 있습니다. (권장: 7.0-10.0)",
            )

            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,  # 더 높은 최대값
                value=35,  # 더 높은 기본값
                step=1,
                label="추론 스텝 수",
                info="이미지 생성 단계 수. 높을수록 품질이 향상되지만 생성 시간이 늘어납니다. (최고 품질: 50-80)",
            )

            sequence_slider = gr.Slider(
                minimum=256,
                maximum=1024,  # 더 높은 최대값
                value=512,  # 더 높은 기본값
                step=64,
                label="최대 시퀀스 길이",
                info="프롬프트 처리 길이. 긴 프롬프트에는 높은 값이 필요합니다. (최고 품질: 512-1024)",
            )

            seed_input = gr.Number(
                label="시드 (-1은 랜덤)",
                value=42,
                precision=0,
                info="생성 결과의 일관성을 위한 난수 시드. 같은 시드로 같은 설정이면 비슷한 결과가 나옵니다. -1은 무작위 시드 사용",
            )

            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

        with gr.Column():
            # 출력 영역
            output_image = gr.Image(label="생성된 이미지", type="pil")
            info_output = gr.Textbox(label="생성 정보", lines=4, interactive=False)

    # 이벤트 연결
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            negative_prompt_input,  # 추가
            input_image,
            guidance_slider,
            steps_slider,
            sequence_slider,
            seed_input,
        ],
        outputs=[output_image, info_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
