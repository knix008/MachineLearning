import gradio as gr
import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
import cv2
import numpy as np
from PIL import Image
import os


# Stable Diffusion 3.5 모델 및 파이프라인 초기화
def initialize_models():
    """Stable Diffusion 3.5 Large와 ControlNet 파이프라인을 초기화합니다."""
    try:
        # SD3 ControlNet 모델 로드 (Canny)
        controlnet = SD3ControlNetModel.from_pretrained(
            "InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float32
        )

        # Stable Diffusion 3.5 Large + ControlNet 파이프라인 로드
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            controlnet=controlnet,
            torch_dtype=torch.float32,
        )

        pipe.enable_model_cpu_offload()
        pipe = pipe.to("cpu")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"모델 로딩 중 오류가 발생했습니다: {e}")
        print("Hugging Face Hub에 로그인했고 모델 접근 권한이 있는지 확인하세요.")
        pipe = None
    return pipe


# Canny edge detection 전처리
def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """입력 이미지에 Canny edge detection을 적용합니다."""
    # PIL Image를 numpy array로 변환
    image_np = np.array(image)

    # RGB를 Grayscale로 변환
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Canny edge detection 적용
    canny = cv2.Canny(gray, low_threshold, high_threshold)

    # 3채널로 변환 (RGB)
    canny_image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(canny_image)


# 이미지 생성 함수 (SD3.5 ControlNet 사용)
def generate_image(
    input_image,
    prompt,
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    controlnet_conditioning_scale=1.0,
    low_threshold=100,
    high_threshold=200,
):
    """Stable Diffusion 3.5 ControlNet을 사용하여 이미지를 생성합니다."""
    try:
        if input_image is None:
            return None, "입력 이미지를 선택해주세요."

        if pipe is None:
            return None, "모델이 로드되지 않았습니다. 모델을 다시 로드해주세요."

        # Canny edge detection 적용
        control_image = preprocess_canny(
            input_image, int(low_threshold), int(high_threshold)
        )

        # 이미지 생성 (ControlNet 사용)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                height=1024,
                width=1024,
            )

        generated_image = result.images[0]
        return generated_image, "이미지 생성 완료!"

    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# 모델 초기화
print("Stable Diffusion 3.5 Large + ControlNet 모델을 초기화하는 중...")
pipe = initialize_models()


# Gradio 인터페이스 생성
def create_interface():
    """Gradio 인터페이스를 생성합니다."""

    with gr.Blocks(
        title="SD 3.5 Large + ControlNet Generator", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 🎨 Stable Diffusion 3.5 Large + ControlNet 이미지 생성기
            
            Stable Diffusion 3.5 Large와 ControlNet을 사용하여 고품질 이미지를 생성합니다.
            입력 이미지의 Canny edge를 정확하게 따라하며 새로운 이미지를 생성할 수 있습니다.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 입력 섹션
                gr.Markdown("### 📥 입력")
                input_image = gr.Image(label="입력 이미지", type="pil", height=300)

                prompt = gr.Textbox(
                    label="프롬프트",
                    value="photorealistic, ultra high definition, ultra high resolution,8k resolution, ultra detail, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, perfect anatomy, good hair, good fingers, good legs",
                    lines=4,
                )

                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트 (선택사항)",
                    value="blurring, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username",
                    lines=4,
                )

                # 고급 설정
                with gr.Accordion("🔧 고급 설정", open=False):
                    num_inference_steps = gr.Slider(
                        minimum=20, maximum=50, value=28, step=1, label="추론 단계 수"
                    )

                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=15.0,
                        value=7.0,
                        step=0.5,
                        label="가이던스 스케일",
                    )

                    controlnet_conditioning_scale = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="ControlNet 컨디셔닝 스케일",
                        visible=True,
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

                generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")

            with gr.Column(scale=1):
                # 출력 섹션
                gr.Markdown("### 📤 출력")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Canny Edge 결과")
                        canny_output = gr.Image(label="Canny Edge", height=250)

                    with gr.Column():
                        gr.Markdown("#### 생성된 이미지")
                        output_image = gr.Image(label="생성 결과", height=250)

                status_text = gr.Textbox(label="상태", interactive=False, lines=2)

        # 예시 프롬프트
        with gr.Row():
            gr.Markdown("### 🖼️ 예시 프롬프트")
            sample_prompts = [
                "a photorealistic portrait of a person",
                "a modern architectural building",
                "a beautiful landscape with mountains",
                "a cute cartoon character",
            ]

            for i, prompt_text in enumerate(sample_prompts):
                gr.Button(f"예시 {i+1}: {prompt_text[:30]}...", size="sm").click(
                    lambda p=prompt_text: p, outputs=[prompt]
                )

        # Canny edge 미리보기 함수
        def preview_canny(image, low_thresh, high_thresh):
            if image is None:
                return None
            return preprocess_canny(image, int(low_thresh), int(high_thresh))

        # 이벤트 핸들러
        def generate_with_canny_preview(*args):
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

            # Canny edge 미리보기 생성
            canny_preview = None
            if input_img is not None:
                canny_preview = preprocess_canny(
                    input_img, int(low_thresh), int(high_thresh)
                )

            # 이미지 생성
            generated_img, status = generate_image(*args)

            return canny_preview, generated_img, status

        # 임계값 변경 시 Canny 미리보기 업데이트
        for component in [input_image, low_threshold, high_threshold]:
            component.change(
                fn=preview_canny,
                inputs=[input_image, low_threshold, high_threshold],
                outputs=[canny_output],
            )

        # 생성 버튼 클릭 이벤트
        generate_btn.click(
            fn=generate_with_canny_preview,
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


# 인터페이스 실행
if __name__ == "__main__":
    demo = create_interface()
    # 인터페이스 실행
    demo.launch(
        share=False, inbrowser=True  # 공유 링크 생성 여부  # 브라우저 자동 열기
    )
