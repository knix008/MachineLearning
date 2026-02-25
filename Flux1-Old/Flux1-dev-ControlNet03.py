import torch
import gradio as gr
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel
from PIL import Image
import cv2
import numpy as np

# 모델 로드
base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

print("모델 로딩 중...")
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

    # 1024보다 큰 경우 비율을 유지하며 축소
    if width > max_size or height > max_size:
        # 더 큰 쪽을 기준으로 축소 비율 계산
        scale_factor = max_size / max(width, height)
        width = int(width * scale_factor)
        height = int(height * scale_factor)

    # 16의 배수로 조정
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16

    # 최소 크기 보장 (256x256)
    new_width = max(new_width, 256)
    new_height = max(new_height, 256)

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

        return output_image, status_message

    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# Gradio 인터페이스 구성
with gr.Blocks(title="FLUX.1 ControlNet 이미지 생성기") as demo:
    gr.Markdown("# FLUX.1 ControlNet 이미지 생성기")
    gr.Markdown(
        "입력 이미지를 업로드하고 프롬프트를 입력하여 새로운 이미지를 생성하세요."
    )
    gr.Markdown(
        "**참고**: 입력 이미지는 자동으로 16의 배수 크기로 조정되며, 1024픽셀보다 큰 경우 원본 비율을 유지하며 축소됩니다."
    )

    with gr.Row():
        with gr.Column(scale=1):
            # 입력 컨트롤
            input_image = gr.Image(
                label="입력 이미지 (1024px 초과시 자동 축소)", type="pil", height=400
            )

            prompt = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="A bohemian-style female wearing a high legged red bikini with sun-kissed skin and messy beach waves.",
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
                    value=24,
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
            output_image = gr.Image(label="생성된 이미지", height=400)
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

    # 파라미터 설명 섹션
    gr.Markdown("## 📖 파라미터 상세 설명")

    with gr.Accordion("🎛️ ControlNet 설정", open=False):
        gr.Markdown(
            """
        ### Canny Edge
        - **용도**: 입력 이미지의 윤곽선을 추출하여 구조적 일관성을 유지
        - **강도 0.0-0.3**: 윤곽선을 참고만 하며 창의적 변형 허용
        - **강도 0.4-0.7**: 균형잡힌 구조 유지와 스타일 변경 (권장)
        - **강도 0.8-1.0**: 윤곽선을 엄격히 준수, 원본 구조 거의 유지
        
        ### Depth Control
        - **용도**: 입력 이미지의 깊이 정보를 활용하여 3D 공간감 유지
        - **강도 0.1-0.3**: 자연스러운 깊이감 유지 (권장)
        - **강도 0.4-0.6**: 뚜렷한 깊이 구조 유지
        - **강도 0.7-1.0**: 원본과 거의 동일한 깊이 구조
        """
        )

    with gr.Accordion("⚙️ 생성 파라미터", open=False):
        gr.Markdown(
            """
        ### 추론 단계 (Inference Steps)
        - **10-15단계**: 빠른 생성, 낮은 품질 (테스트용)
        - **20-30단계**: 균형잡힌 품질과 속도 (일반 사용 권장)
        - **35-50단계**: 고품질 생성, 느린 속도 (최종 결과물용)
        
        ### 가이던스 스케일 (Guidance Scale)
        - **1.0-3.0**: 창의적이고 예술적인 결과, 프롬프트에서 벗어날 수 있음
        - **3.5-7.0**: 프롬프트와 창의성의 균형 (권장)
        - **7.5-10.0**: 프롬프트에 매우 충실, 덜 창의적
        
        ### 시드 (Seed)
        - **고정 시드**: 같은 설정에서 동일한 결과 재현 가능
        - **랜덤 시드 (-1)**: 매번 다른 결과, 다양한 변형 탐색
        """
        )

    with gr.Accordion("📐 이미지 크기 처리", open=False):
        gr.Markdown(
            """
        ### 크기 조정 규칙
        - **1024px 이하**: 원본 크기 유지 후 16의 배수로 조정
        - **1024px 초과**: 긴 쪽을 1024px로 축소하며 비율 유지 후 16의 배수로 조정
        - **최소 크기**: 256x256 보장
        
        ### 예시
        - **800x600** → 800x592 (16의 배수로 조정)
        - **2048x1536** → 1024x768 (축소 후 16의 배수로 조정)
        - **100x200** → 256x256 (최소 크기 보장)
        """
        )

    with gr.Accordion("💡 사용 팁", open=False):
        gr.Markdown(
            """
        ### 🎯 목적별 권장 설정
        
        **포트레이트 변환**
        - Canny 강도: 0.5-0.7 (얼굴 구조 유지)
        - 추론 단계: 25-35
        - 가이던스: 4-6
        
        **풍경 스타일 변환**
        - Canny 강도: 0.3-0.5 (자연스러운 변형)
        - Depth 강도: 0.2-0.4 (공간감 유지)
        - 추론 단계: 20-30
        - 가이던스: 3-5
        
        **예술적 스타일 변환**
        - Canny 강도: 0.2-0.4 (창의적 변형 허용)
        - 추론 단계: 30-40
        - 가이던스: 2-4
        
        **사실적 변환**
        - Canny 강도: 0.6-0.8 (구조 엄격히 유지)
        - Depth 강도: 0.3-0.5
        - 추론 단계: 35-50
        - 가이던스: 5-7
        
        ### 🚀 성능 최적화
        - **빠른 테스트**: 추론 단계 15, 가이던스 3.5
        - **균형잡힌 품질**: 추론 단계 24, 가이던스 3.5 (기본값)
        - **최고 품질**: 추론 단계 40, 가이던스 5.0
        """
        )

    # 사용 방법
    gr.Markdown("## 🎨 사용 방법")
    gr.Markdown(
        """
    1. **이미지 업로드**: 변환하고 싶은 이미지를 업로드하세요 (고해상도 이미지는 자동으로 축소됩니다)
    2. **프롬프트 작성**: 원하는 스타일이나 내용을 상세히 설명하세요
    3. **ControlNet 설정**: 구조 유지 정도를 조정하세요
    4. **생성 파라미터**: 품질과 속도의 균형을 맞춰 설정하세요
    5. **생성 실행**: '이미지 생성' 버튼을 클릭하세요
    
    **💡 팁**: 고해상도 이미지도 부담 없이 업로드하세요. 자동으로 최적 크기로 조정됩니다!
    """
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
