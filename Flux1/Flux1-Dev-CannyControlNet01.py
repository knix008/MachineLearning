import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import gradio as gr
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from PIL import Image
import cv2
import numpy as np
import datetime

# 모델 로드
base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

controlnet_union = FluxControlNetModel.from_pretrained(
    controlnet_model_union, torch_dtype=torch.bfloat16
)
controlnet = FluxMultiControlNetModel([controlnet_union])

pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)

# 메모리 최적화
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
print("모델 로딩 완료!")


def resize_to_multiple_of_16(image, max_size=1024):
    """이미지를 16의 배수로 리사이즈하면서 비율 유지"""
    width, height = image.size

    # 16의 배수로 조정
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16

    # 이미지 리사이즈
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, new_width, new_height


def create_canny_edge(image, low_threshold=50, high_threshold=150):
    """입력 이미지에서 Canny edge 이미지 생성"""
    image_array = np.array(image)
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    canny = cv2.Canny(gray, low_threshold, high_threshold)
    canny_image = Image.fromarray(canny).convert("RGB")
    return canny_image


def generate_image(
    prompt,
    input_image,
    use_canny=True,
    use_depth=False,
    canny_strength=0.4,
    depth_strength=0.2,
    num_steps=24,
    guidance_scale=3.5,
    seed=42,
):
    """이미지 생성 함수"""
    if input_image is None:
        return None, "입력 이미지를 업로드해주세요."

    try:
        # 입력 이미지 전처리 - RGB 변환 및 16의 배수로 리사이즈
        input_image = input_image.convert("RGB")
        original_size = input_image.size

        # 16의 배수로 리사이즈 (1024보다 큰 경우 축소)
        resized_image, width, height = resize_to_multiple_of_16(input_image)

        control_images = []
        control_modes = []
        conditioning_scales = []

        if use_canny:
            # Canny edge 이미지 생성 (리사이즈된 이미지 사용)
            canny_image = create_canny_edge(resized_image)
            control_images.append(canny_image)
            control_modes.append(0)  # Canny mode
            conditioning_scales.append(canny_strength)

        if use_depth:
            # 여기서는 리사이즈된 이미지를 depth로 사용 (실제로는 depth 모델이 필요)
            control_images.append(resized_image)
            control_modes.append(2)  # Depth mode
            conditioning_scales.append(depth_strength)

        if not control_images:
            return None, "최소 하나의 컨트롤 모드를 선택해주세요."

        # 이미지 생성
        generator = torch.manual_seed(seed) if seed != -1 else None

        result = pipe(
            prompt=prompt,
            control_image=control_images,
            control_mode=control_modes,
            width=width,
            height=height,
            controlnet_conditioning_scale=conditioning_scales,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        output_image = result.images[0]

        # 축소 여부 확인하여 상태 메시지 작성
        was_resized = original_size[0] > 1024 or original_size[1] > 1024
        if was_resized:
            status_message = f"이미지 생성 완료! 원본 크기: {original_size[0]}x{original_size[1]} → 축소 및 조정된 크기: {width}x{height}"
        else:
            status_message = f"이미지 생성 완료! 원본 크기: {original_size[0]}x{original_size[1]} → 조정된 크기: {width}x{height}"

        output_image.save(
            f"flux1-dev-canny-controlnet01_({datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}).png"
        )

        return output_image, status_message

    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# Gradio 인터페이스 구성
with gr.Blocks(title="FLUX.1 Canny Union Pro 이미지 생성기") as demo:
    gr.Markdown("# FLUX.1 Canny Union Pro 이미지 생성기")

    with gr.Row():
        with gr.Column(scale=1):
            # 입력 컨트롤
            input_image = gr.Image(
                label="입력 이미지", type="pil", height=500, value="default.jpg"
            )

            prompt = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="8k, high detail, realistic, high quality, masterpiece, best quality, dark blue bikini",
                lines=3,
                info="생성하고 싶은 이미지의 스타일, 내용, 분위기 등을 상세히 설명하세요. 영어로 작성하는 것이 더 효과적입니다.",
            )

            with gr.Row():
                use_canny = gr.Checkbox(
                    label="Canny Edge 사용",
                    value=True,
                    info="입력 이미지의 윤곽선을 추출하여 구조를 유지하면서 새 이미지를 생성합니다.",
                )
                use_depth = gr.Checkbox(
                    label="Depth 사용",
                    value=False,
                    info="입력 이미지의 깊이 정보를 활용하여 공간적 구조를 유지합니다.",
                )

            with gr.Row():
                canny_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.4,
                    step=0.1,
                    label="Canny 강도",
                    info="0.0: 윤곽선 무시, 1.0: 윤곽선 완전 준수. 권장값: 0.3-0.7",
                )
                depth_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    label="Depth 강도",
                    info="0.0: 깊이 정보 무시, 1.0: 깊이 완전 준수. 권장값: 0.1-0.5",
                )

            with gr.Row():
                num_steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=28,
                    step=1,
                    label="추론 단계",
                    info="많을수록 고품질이지만 시간이 오래 걸림. 빠른 생성: 10-20, 고품질: 30-50",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.5,
                    label="가이던스 스케일",
                    info="프롬프트 충실도. 낮음: 창의적, 높음: 프롬프트 준수. 권장값: 3-7",
                )

            seed = gr.Number(
                label="시드 (-1은 랜덤)",
                value=42,
                precision=0,
                info="같은 시드는 동일한 결과를 생성합니다. -1로 설정하면 매번 다른 결과가 나옵니다.",
            )

            generate_btn = gr.Button("이미지 생성", variant="primary")

        with gr.Column(scale=1):
            # 출력
            output_image = gr.Image(label="생성된 이미지", height=500)
            status_text = gr.Textbox(label="상태 (크기 정보 포함)", interactive=False)

    # 이벤트 핸들러
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            input_image,
            use_canny,
            use_depth,
            canny_strength,
            depth_strength,
            num_steps,
            guidance_scale,
            seed,
        ],
        outputs=[output_image, status_text],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
