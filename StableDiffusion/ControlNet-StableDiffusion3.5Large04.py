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


# Canny edge detection 전처리 함수
def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """입력 이미지에 Canny edge detection을 적용합니다."""
    if image is None:
        return None

    # 이미지 크기를 16의 배수로 조정
    width, height = image.size

    # 16의 배수로 조정 (가장 가까운 16의 배수로 반올림)
    new_width = ((width + 8) // 16) * 16
    new_height = ((height + 8) // 16) * 16

    # 최소 크기 보장 (512x512)
    new_width = max(512, new_width)
    new_height = max(512, new_height)

    # 최대 크기 제한 (1536x1536)
    new_width = min(1536, new_width)
    new_height = min(1536, new_height)

    # 크기가 변경되었으면 리사이즈
    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(
            f"이미지 크기가 16의 배수로 조정되었습니다: {width}x{height} -> {new_width}x{new_height}"
        )

    # OpenCV를 사용하여 Canny Edge를 검출합니다.
    image_np = np.array(image)
    canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)

    return Image.fromarray(canny_image)


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
                    value="A beautiful woman wearing a red bikini and walking on the beach",
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
                gr.Button(prompt_text, size="sm").click(
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

            # 이미지 생성
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
    print("🚀 Stable Diffusion 3.5 Large + ControlNet GUI를 시작합니다...")

    # 모델 자동 초기화
    print("모델을 초기화하는 중...")
    success = initialize_models()
    if not success:
        print("❌ 모델 초기화에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)

    # Gradio 인터페이스 생성 및 실행
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)
