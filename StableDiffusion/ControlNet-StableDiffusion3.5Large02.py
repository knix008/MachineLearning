import torch
import gradio as gr
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
import cv2
import numpy as np
from PIL import Image

# 전역 변수로 파이프라인 저장
pipe = None


# 모델 초기화 함수
def initialize_models():
    """Stable Diffusion 3.5 Large ControlNet 모델을 초기화합니다."""
    global pipe
    try:
        print("ControlNet 모델을 로딩하는 중...")
        # SD3 ControlNet 모델 로드
        controlnet = SD3ControlNetModel.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
            torch_dtype=torch.float16,
        )

        print("Stable Diffusion 3.5 Large 파이프라인을 로딩하는 중...")
        # SD 3.5 Large 파이프라인 로드
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )

        # GPU 사용 가능한 경우 GPU로, 아니면 CPU로
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)

        print(f"모델이 {device}에 성공적으로 로드되었습니다!")
        return True
    except Exception as e:
        print(f"모델 로딩 중 오류가 발생했습니다: {e}")
        return False


# 해상도 정규화 함수
def normalize_resolution(width, height, base=16):
    """해상도를 지정된 배수로 조정하면서 원본 비율을 최대한 유지합니다."""
    # 원본 비율 계산
    aspect_ratio = width / height

    # 16의 배수로 반올림
    new_width = ((width + base // 2) // base) * base
    new_height = ((height + base // 2) // base) * base

    # 최소/최대 해상도 제한 (비율 유지하면서)
    min_res = 512
    max_res = 1536

    # 크기가 범위를 벗어나는 경우 비율을 유지하면서 조정
    if new_width > max_res or new_height > max_res:
        if new_width > new_height:
            # 가로가 더 긴 경우
            scale = max_res / new_width
            new_width = max_res
            new_height = int((new_height * scale + base // 2) // base) * base
        else:
            # 세로가 더 긴 경우
            scale = max_res / new_height
            new_height = max_res
            new_width = int((new_width * scale + base // 2) // base) * base

    if new_width < min_res or new_height < min_res:
        if new_width < new_height:
            # 가로가 더 짧은 경우
            scale = min_res / new_width
            new_width = min_res
            new_height = int((new_height * scale + base // 2) // base) * base
        else:
            # 세로가 더 짧은 경우
            scale = min_res / new_height
            new_height = min_res
            new_width = int((new_width * scale + base // 2) // base) * base

    # 최종 16의 배수 보장
    new_width = (new_width // base) * base
    new_height = (new_height // base) * base

    # 최소값 재확인
    new_width = max(min_res, new_width)
    new_height = max(min_res, new_height)

    return new_width, new_height


# Canny edge detection 전처리 함수
def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """입력 이미지에 Canny edge detection을 적용합니다."""
    if image is None:
        return None

    try:
        # 이미지 크기 제약 조건 확인 및 비율 유지 리사이즈
        width, height = image.size
        original_aspect_ratio = width / height
        max_size = 1536  # 최대 해상도 제한 (2048에서 줄임)
        min_size = 512  # 최소 해상도 제한

        # 크기 조정이 필요한 경우 비율을 유지하면서 조정
        if width > max_size or height > max_size:
            # 비율을 유지하면서 최대 크기에 맞춤
            if width > height:
                new_width = max_size
                new_height = int(max_size / original_aspect_ratio)
            else:
                new_height = max_size
                new_width = int(max_size * original_aspect_ratio)

            # 16의 배수로 조정
            new_width, new_height = normalize_resolution(new_width, new_height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(
                f"이미지 크기가 비율 유지하며 조정되었습니다: {width}x{height} -> {new_width}x{new_height}"
            )
            print(
                f"원본 비율: {original_aspect_ratio:.3f}, 조정 후 비율: {new_width/new_height:.3f}"
            )
            width, height = new_width, new_height

        elif width < min_size and height < min_size:
            # 비율을 유지하면서 최소 크기에 맞춤
            if width > height:
                new_height = min_size
                new_width = int(min_size * original_aspect_ratio)
            else:
                new_width = min_size
                new_height = int(min_size / original_aspect_ratio)

            # 16의 배수로 조정
            new_width, new_height = normalize_resolution(new_width, new_height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(
                f"이미지 크기가 비율 유지하며 확대되었습니다: {width}x{height} -> {new_width}x{new_height}"
            )
            print(
                f"원본 비율: {original_aspect_ratio:.3f}, 조정 후 비율: {new_width/new_height:.3f}"
            )
            width, height = new_width, new_height

        else:
            # 16의 배수로만 조정 (비율 최소 변화)
            normalized_width, normalized_height = normalize_resolution(width, height)
            if (normalized_width, normalized_height) != (width, height):
                image = image.resize(
                    (normalized_width, normalized_height), Image.Resampling.LANCZOS
                )
                print(
                    f"해상도가 16의 배수로 조정되었습니다: {width}x{height} -> {normalized_width}x{normalized_height}"
                )
                print(
                    f"원본 비율: {original_aspect_ratio:.3f}, 조정 후 비율: {normalized_width/normalized_height:.3f}"
                )
                width, height = normalized_width, normalized_height

        # PIL Image를 numpy array로 변환
        image_np = np.array(image)

        # 이미지 채널 확인 및 처리
        if len(image_np.shape) == 4:  # RGBA
            # 알파 채널 제거하고 RGB로 변환
            image_np = image_np[:, :, :3]
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # RGB
            pass  # 그대로 사용
        elif (
            len(image_np.shape) == 3 and image_np.shape[2] == 1
        ):  # Grayscale with 1 channel
            image_np = image_np.squeeze()
        elif len(image_np.shape) == 2:  # Pure Grayscale
            pass  # 그대로 사용
        else:
            raise ValueError(f"지원되지 않는 이미지 형식: {image_np.shape}")

        # RGB를 Grayscale로 변환
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Canny edge detection 적용
        canny = cv2.Canny(gray, low_threshold, high_threshold)

        # 3채널로 변환 (RGB)
        canny_image = canny[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

        return Image.fromarray(canny_image)

    except Exception as e:
        print(f"이미지 전처리 중 오류 발생: {e}")
        return None


# 이미지 생성 함수
def generate_image(
    input_image,
    prompt,
    negative_prompt="",
    num_inference_steps=25,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
    low_threshold=100,
    high_threshold=200,
):
    """SD 3.5 Large ControlNet을 사용하여 이미지를 생성합니다."""
    global pipe

    try:
        if pipe is None:
            return None, "모델이 로드되지 않았습니다. 먼저 모델을 초기화해주세요."

        if input_image is None:
            return None, "입력 이미지를 선택해주세요."

        # 이미지 형식 및 크기 검증
        try:
            width, height = input_image.size
            if width * height > 1536 * 1536:  # 최대 해상도 변경
                return (
                    None,
                    "이미지가 너무 큽니다. 1536x1536 이하의 이미지를 사용해주세요.",
                )
            if width < 256 or height < 256:
                return (
                    None,
                    "이미지가 너무 작습니다. 최소 256x256 크기의 이미지를 사용해주세요.",
                )
        except Exception as e:
            return None, f"이미지 형식이 올바르지 않습니다: {str(e)}"

        if not prompt.strip():
            return None, "프롬프트를 입력해주세요."

        # Canny edge detection 적용
        control_image = preprocess_canny(
            input_image, int(low_threshold), int(high_threshold)
        )

        if control_image is None:
            return None, "Canny edge 처리 중 오류가 발생했습니다."

        # 제어 이미지의 해상도를 출력 해상도로 사용
        output_width, output_height = control_image.size
        print(f"출력 해상도: {output_width}x{output_height}")

        # 이미지 생성
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                control_image=control_image,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                height=output_height,
                width=output_width,
            )

        generated_image = result.images[0]
        return generated_image, "이미지 생성 완료!"

    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# Gradio 인터페이스 생성
def create_interface():
    """Gradio 인터페이스를 생성합니다."""

    with gr.Blocks(
        title="SD 3.5 Large ControlNet Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-container { max-width: 1200px; margin: 0 auto; }
        .status-box { background-color: #f0f8ff; padding: 10px; border-radius: 5px; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🎨 Stable Diffusion 3.5 Large + ControlNet 이미지 생성기
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 입력 섹션
                gr.Markdown("### 📥 입력")
                input_image = gr.Image(
                    label="입력 이미지 (구조 참조용)", type="pil", height=350
                )

                prompt = gr.Textbox(
                    label="프롬프트",
                    placeholder="예: A beautiful landscape painting in impressionist style",
                    lines=3,
                    value="A photorealistic, high-quality, detailed, masterpiece, 8k resolution, professional photography, sharp focus, vivid colors, perfect composition, dramatic lighting, cinematic quality",
                )

                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트 (선택사항)",
                    placeholder="예: blurry, low quality, distorted",
                    lines=2,
                    value="blurry, low quality, bad anatomy, distorted, ugly, deformed, poorly drawn, bad hands, bad fingers, missing limbs, extra limbs, cropped, worst quality, low resolution, jpeg artifacts, watermark, text, signature, username, over saturated, under saturated, overexposed, underexposed",
                )

                # 고급 설정
                with gr.Accordion("🔧 고급 설정", open=False):
                    num_inference_steps = gr.Slider(
                        minimum=15,
                        maximum=50,
                        value=25,
                        step=1,
                        label="추론 단계 수 (높을수록 품질 향상, 시간 증가)",
                    )

                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        label="가이던스 스케일 (프롬프트 준수도)",
                    )

                    controlnet_conditioning_scale = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="ControlNet 강도 (구조 따라하기 정도)",
                    )

                    with gr.Row():
                        low_threshold = gr.Slider(
                            minimum=50,
                            maximum=150,
                            value=100,
                            step=10,
                            label="Canny 낮은 임계값",
                        )

                        high_threshold = gr.Slider(
                            minimum=150,
                            maximum=300,
                            value=200,
                            step=10,
                            label="Canny 높은 임계값",
                        )

                generate_btn = gr.Button(
                    "🎨 이미지 생성", variant="primary", size="lg", interactive=True
                )

            with gr.Column(scale=1):
                # 출력 섹션
                gr.Markdown("### 📤 출력")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Canny Edge 미리보기")
                        canny_output = gr.Image(label="Canny Edge", height=300)

                    with gr.Column():
                        gr.Markdown("#### 생성된 이미지")
                        output_image = gr.Image(label="생성 결과", height=300)

                status_text = gr.Textbox(label="상태", interactive=False, lines=2)

        # 예시 프롬프트 버튼들
        with gr.Row():
            gr.Markdown("### 🎯 예시 프롬프트")

        with gr.Row():
            example_prompts = [
                "A majestic lion in a savanna, photorealistic, golden hour lighting",
                "A futuristic cyberpunk cityscape at night, neon lights, detailed",
                "A serene mountain landscape, oil painting style, peaceful atmosphere",
                "An elegant portrait of a woman, renaissance painting style, detailed",
            ]

            for i, prompt_text in enumerate(example_prompts):
                gr.Button(f"예시 {i+1}", size="sm").click(
                    lambda p=prompt_text: p, outputs=[prompt]
                )

        # 이벤트 함수들
        def preview_canny(image, low_thresh, high_thresh):
            if image is None:
                return None
            return preprocess_canny(image, int(low_thresh), int(high_thresh))

        def generate_with_preview(*args):
            (
                input_img,
                prompt_text,
                neg_prompt,
                steps,
                guid_scale,
                ctrl_scale,
                low_thresh,
                high_thresh,
            ) = args

            # Canny 미리보기 생성
            canny_preview = None
            if input_img is not None:
                canny_preview = preprocess_canny(
                    input_img, int(low_thresh), int(high_thresh)
                )

            # 이미지 생성 - 올바른 인수 순서로 호출
            generated_img, status = generate_image(
                input_image=input_img,
                prompt=prompt_text,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=guid_scale,
                controlnet_conditioning_scale=ctrl_scale,
                low_threshold=low_thresh,
                high_threshold=high_thresh,
            )

            return canny_preview, generated_img, status

        # 이벤트 바인딩
        # Canny 미리보기 업데이트
        for component in [input_image, low_threshold, high_threshold]:
            component.change(
                fn=preview_canny,
                inputs=[input_image, low_threshold, high_threshold],
                outputs=[canny_output],
            )

        # 이미지 생성
        generate_btn.click(
            fn=generate_with_preview,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                controlnet_conditioning_scale,
                low_threshold,
                high_threshold,
            ],
            outputs=[canny_output, output_image, status_text],
        )

    return demo


# 메인 실행부
if __name__ == "__main__":
    print("🚀 Stable Diffusion 3.5 Large ControlNet GUI를 시작합니다...")

    # 모델 자동 초기화
    print("모델을 초기화하는 중...")
    success = initialize_models()
    if not success:
        print("❌ 모델 초기화에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)

    # Gradio 인터페이스 생성 및 실행
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)
