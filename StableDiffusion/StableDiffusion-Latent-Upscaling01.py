import torch
import gradio as gr
from diffusers import StableDiffusionLatentUpscalePipeline
from PIL import Image
import datetime

# 모델 로드
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
    "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16, use_safetensors=True
)
upscaler.enable_model_cpu_offload()
upscaler.enable_xformers_memory_efficient_attention()


def preprocess_image(image, max_size):
    """
    이미지를 모델에 적합한 크기로 전처리 (비율 유지하며 64의 배수로 조정)
    """
    width, height = image.size
    
    # 원본 비율 계산
    aspect_ratio = width / height
    
    # 최대 크기를 넘지 않도록 조정
    if width > height:
        # 가로가 더 긴 경우
        new_width = min(max_size, width)
        new_height = int(new_width / aspect_ratio)
    else:
        # 세로가 더 긴 경우
        new_height = min(max_size, height)
        new_width = int(new_height * aspect_ratio)
    
    # 64의 배수로 맞추기 (더 안전한 방법)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    # 최소 크기 확보
    min_size = 256
    if new_width == 0 or new_width < min_size:
        new_width = 256
    if new_height == 0 or new_height < min_size:
        new_height = 256
    
    # 다시 64의 배수로 보정
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    # 최종 안전 검증
    if new_width == 0:
        new_width = 256
    if new_height == 0:
        new_height = 256
    
    print(f"원본 크기: {width}x{height} -> 전처리된 크기: {new_width}x{new_height}")
    print(f"비율 유지 확인: 원본 비율 {aspect_ratio:.3f}, 처리된 비율 {new_width/new_height:.3f}")
    
    # 리사이즈
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image


def upscale_image(
    input_image,
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    max_size,
    seed,
):
    if input_image is None:
        return None

    try:
        # 입력 이미지가 numpy array인 경우 PIL Image로 변환
        if hasattr(input_image, "shape"):
            input_image = Image.fromarray(input_image)

        # 이미지 전처리
        processed_image = preprocess_image(input_image, max_size)

        # 프롬프트가 비어있으면 기본값 사용
        if not prompt.strip():
            prompt = "high quality, detailed"

        # 시드 설정
        generator = None
        if seed >= 0:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        # 업스케일링 수행
        upscaled_image = upscaler(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            image=processed_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        upscaled_image.save(f"stablediffusion_upscaled_image-({datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}).png")

        return upscaled_image

    except Exception as e:
        print(f"업스케일링 중 오류 발생: {str(e)}")
        return None


# Gradio 인터페이스 생성
with gr.Blocks(title="Stable Diffusion Latent Upscaler") as demo:
    gr.Markdown("# Stable Diffusion Latent Upscaler")
    gr.Markdown("이미지와 프롬프트를 입력하면 2배 업스케일된 이미지를 생성합니다.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="입력 이미지",
                type="pil",
                sources=["upload", "clipboard"],
                value="default.jpg",
                height=500,
            )

            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="예: high quality, detailed, 8k",
                value="8k, high detail, high quality, photo realistic, masterpiece, best quality, dark blue bikini, skinny",
                lines=3,
            )

            negative_prompt_input = gr.Textbox(
                label="네거티브 프롬프트 (선택사항)",
                placeholder="예: blurry, low quality, artifacts",
                value="blurring, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, normal quality, jpeg artifacts, signature, watermark, username",
                lines=2,
            )

            with gr.Accordion("고급 설정", open=False):
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=50,
                    step=1,
                    label="추론 단계 수 (Inference Steps)",
                    info="높을수록 품질이 좋아지지만 시간이 오래 걸립니다. 일반적으로 15-25가 적당합니다.",
                )

                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=19.0,
                    value=3.5,
                    step=0.1,
                    label="가이던스 스케일 (Guidance Scale)",
                    info="프롬프트를 얼마나 강하게 따를지 결정합니다. Latent Upscaler는 0에 가까운 값이 좋습니다.",
                )

                max_size = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="최대 이미지 크기",
                    info="메모리 제한을 위한 최대 크기입니다. GPU 메모리에 따라 조정하세요.",
                )

                seed = gr.Number(
                    label="시드 (Seed)",
                    value=42,
                    precision=0,
                    info="재현 가능한 결과를 위한 시드값입니다. -1은 랜덤 시드를 의미합니다.",
                )

            upscale_btn = gr.Button("업스케일", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="업스케일된 이미지", height=500, type="pil")

            # 출력 이미지 정보와 다운로드 버튼
            with gr.Row():
                output_info = gr.Textbox(label="출력 이미지 정보", interactive=False, scale=2)
                download_btn = gr.DownloadButton(
                    label="이미지 다운로드",
                    variant="secondary",
                    size="sm",
                    scale=1,
                    visible=False
                )

    def update_image_info(output_img):
        output_text = ""
        download_file = None
        download_visible = False

        if output_img is not None:
            # PIL Image인지 확인
            if hasattr(output_img, 'size') and isinstance(output_img.size, tuple):
                output_text = f"크기: {output_img.size[0]}x{output_img.size[1]}"

                # 임시 파일로 저장하여 다운로드 가능하게 만들기
                temp_filename = f"upscaled_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                output_img.save(temp_filename)
                download_file = temp_filename
                download_visible = True
            else:
                output_text = "출력 이미지 정보를 읽을 수 없습니다."

        return output_text, gr.DownloadButton(
            label="이미지 다운로드",
            value=download_file,
            variant="secondary",
            size="sm",
            visible=download_visible
        )

    # 버튼 클릭 이벤트
    upscale_result = upscale_btn.click(
        fn=upscale_image,
        inputs=[
            input_image,
            prompt_input,
            negative_prompt_input,
            num_inference_steps,
            guidance_scale,
            max_size,
            seed,
        ],
        outputs=output_image,
    )

    # 출력 이미지 정보 업데이트 및 다운로드 버튼 활성화
    upscale_result.then(
        fn=update_image_info,
        inputs=[output_image],
        outputs=[output_info, download_btn],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False)
