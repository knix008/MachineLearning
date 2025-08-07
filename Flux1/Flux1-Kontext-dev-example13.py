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


def resize_image(input_image):
    # 입력 이미지 크기
    w, h = input_image.size
    print(f"원본 입력 이미지 크기: {w}x{h}")

    # 16으로 나누어떨어지도록 크기 조정
    def round_to_multiple(value, multiple=16):
        return ((value + multiple // 2) // multiple) * multiple

    # 원본 비율 계산
    aspect_ratio = w / h
    
    # 최대 크기 제한 (1360)
    max_size = 1360
    
    # 더 큰 쪽이 1360을 넘는 경우 비율을 유지하며 축소
    if max(w, h) > max_size:
        if w >= h:
            # 가로가 더 큰 경우
            scale_factor = max_size / w
            w = max_size
            h = int(h * scale_factor)
        else:
            # 세로가 더 큰 경우
            scale_factor = max_size / h
            h = max_size
            w = int(w * scale_factor)
        print(f"최대 크기 제한 적용 후: {w}x{h}")

    # 16으로 나누어떨어지도록 크기 조정
    if w >= h:
        # 가로가 더 크거나 같은 경우
        new_w = round_to_multiple(w)
        new_h = round_to_multiple(int(new_w / aspect_ratio))
        
        # 새로운 크기가 제한을 넘는 경우 다시 조정
        if new_w > max_size:
            new_w = round_to_multiple(max_size - 16)  # 한 단계 작게
            new_h = round_to_multiple(int(new_w / aspect_ratio))
    else:
        # 세로가 더 큰 경우
        new_h = round_to_multiple(h)
        new_w = round_to_multiple(int(new_h * aspect_ratio))
        
        # 새로운 크기가 제한을 넘는 경우 다시 조정
        if new_h > max_size:
            new_h = round_to_multiple(max_size - 16)  # 한 단계 작게
            new_w = round_to_multiple(int(new_h * aspect_ratio))

    # 비율이 많이 틀어지지 않도록 재조정
    actual_ratio = new_w / new_h
    if abs(actual_ratio - aspect_ratio) > 0.1:  # 비율 차이가 10% 이상인 경우
        if aspect_ratio > actual_ratio:
            new_w = min(round_to_multiple(int(new_h * aspect_ratio)), max_size)
        else:
            new_h = min(round_to_multiple(int(new_w / aspect_ratio)), max_size)

    print(f"조정된 이미지 크기: {new_w}x{new_h}")
    print(f"원본 비율: {aspect_ratio:.3f}, 조정된 비율: {new_w/new_h:.3f}")

    # 이미지 리사이즈 (고품질 리샘플링 사용)
    resized_image = input_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

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
        filename = f"flux1-kontext-dev-example13_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
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
                value="default.jpg",
            )

            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="8k, high detail, high quality, realistic, masterpiece, best quality, dark blue bikini",
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
                value=6.5,  # 더 높은 기본값
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
