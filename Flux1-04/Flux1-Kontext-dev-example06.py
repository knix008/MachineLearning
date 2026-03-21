import torch
import gradio as gr
import time
from diffusers import FluxKontextPipeline
from PIL import Image

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
    guidance_scale,
    num_inference_steps,
    max_sequence_length,
    seed,
    style_preset_name,  # Add this parameter
    progress=gr.Progress(),
):
    """이미지 생성 함수 (FluxKontext 전용 - Image-to-Image만 지원)"""

    # ⭐ 입력 이미지 필수 체크
    if input_image is None:
        error_text = "❌ FluxKontext는 Image-to-Image 전용 모델입니다.\n\n📸 변환할 이미지를 업로드해주세요!\n\n사용법:\n1. 좌측 상단에 이미지를 업로드하세요\n2. 스타일 프리셋을 선택하세요\n3. 필요시 프롬프트를 수정하세요\n4. '🎨 이미지 생성' 버튼을 클릭하세요"
        return None, error_text

    start_time = time.time()

    # Progress bar 시작
    progress(0.1, desc="🎨 이미지 처리 시작...")

    # ⭐ 선택된 스타일 프리셋의 프롬프트 적용
    preset = STYLE_PRESETS.get(style_preset_name, STYLE_PRESETS["기본"])
    preset_prompt = preset["prompt_prefix"]
    
    # 사용자 프롬프트와 프리셋 프롬프트 결합
    if prompt.strip():
        final_prompt = f"{preset_prompt}, {prompt.strip()}"
    else:
        final_prompt = preset_prompt

    # ⭐ 원본 이미지 크기 자동 사용
    original_width, original_height = input_image.size
    original_ratio = original_width / original_height

    progress(0.2, desc=f"📐 원본 크기 감지: {original_width}x{original_height}")

    # 시드 설정
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator("cpu").manual_seed(seed)

    try:
        progress(0.3, desc="🖼️ 입력 이미지 전처리 중...")

        # ⭐ 무조건 원본 크기 그대로 사용
        input_image_for_processing = input_image.convert("RGB")
        progress(0.5, desc="🧠 AI 모델 처리 중 (원본 크기 유지)...")
        
        print("> ⭐ Input to Pipeline ⭐")
        print("===========================================================")
        print("> Final Prompt : ", final_prompt)  # Changed to show final prompt
        print("> Preset Used : ", style_preset_name)
        print("> User Input : ", prompt)
        print("> Seed : ", seed)
        print("> Guidance Scale : ", guidance_scale)
        print("> Num Inference Steps : ", num_inference_steps)
        print("> Max Sequence Length : ", max_sequence_length)
        print("> =========================================================")

        # ⭐ 원본 크기 그대로 모델에 전달 (final_prompt 사용)
        image = pipe(
            prompt=final_prompt,  # Use the combined prompt
            image=input_image_for_processing,
            width=original_width,
            height=original_height,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).images[0]

        progress(0.9, desc="💾 결과 저장 중...")

        end_time = time.time()
        generation_time = end_time - start_time

        # 이미지 저장 (고품질)
        timestamp = int(time.time())
        filename = f"flux_enhanced_{timestamp}.png"
        image.save(filename, optimize=True, quality=95)

        # 생성된 이미지 크기 정보
        generated_width, generated_height = image.size
        final_ratio = generated_width / generated_height

        # ⭐ 크기 유지 정보
        size_info = f"\n📐 이미지 크기: {original_width}x{original_height} → {generated_width}x{generated_height}"
        size_info += f"\n📏 비율 유지: {original_ratio:.3f} → {final_ratio:.3f} (차이: {abs(original_ratio - final_ratio):.3f})"
        
        # 크기 변화 정보
        if generated_width == original_width and generated_height == original_height:
            size_info += f"\n✅ 원본 크기 완전 유지"
        else:
            size_change_w = abs(original_width - generated_width)
            size_change_h = abs(original_height - generated_height)
            size_info += f"\n📏 크기 변화: 가로 {size_change_w}px, 세로 {size_change_h}px"

        progress(1.0, desc="✅ 완료!")

        info_text = f"✅ 이미지 변환 완료!\n⏱️ 처리 시간: {generation_time:.2f}초\n🎲 시드: {seed}\n💾 저장 파일: {filename}\n🎨 가이던스: {guidance_scale} | ⚡ 스텝: {num_inference_steps}{size_info}\n🎯 사용된 프리셋: {style_preset_name}\n\n📝 최종 프롬프트:\n{final_prompt[:150]}..."  # Show final prompt in info

        return image, info_text

    except Exception as e:
        progress(1.0, desc="❌ 오류 발생")
        error_text = f"❌ 오류 발생: {str(e)}\n\n💡 해결 방법:\n- 다른 스타일 프리셋을 시도해보세요\n- 이미지 파일 형식을 확인해주세요\n- 프롬프트를 더 간단하게 작성해보세요"
        return None, error_text


# 프리셋 선택 시 자동 설정 적용
def apply_preset_settings(preset_name):
    """선택한 프리셋에 따라 설정값 자동 조정"""
    preset = STYLE_PRESETS.get(preset_name, STYLE_PRESETS["기본"])
    return preset["guidance_scale"], preset["num_inference_steps"]


# ⭐ 이미지 비율 분석 유틸리티 함수들
def get_aspect_ratio_info(image):
    """이미지의 비율 정보를 분석하여 반환"""
    if image is None:
        return "이미지가 없습니다."
    
    w, h = image.size
    ratio = w / h
    
    # 일반적인 비율 매칭
    if abs(ratio - 1.0) < 0.05:
        ratio_name = "정사각형 (1:1)"
    elif abs(ratio - 4/3) < 0.05:
        ratio_name = "표준 (4:3)"
    elif abs(ratio - 16/9) < 0.05:
        ratio_name = "와이드 (16:9)"
    elif abs(ratio - 3/2) < 0.05:
        ratio_name = "사진 (3:2)"
    elif abs(ratio - 9/16) < 0.05:
        ratio_name = "세로형 (9:16)"
    elif abs(ratio - 2/3) < 0.05:
        ratio_name = "세로형 (2:3)"
    elif abs(ratio - 5/4) < 0.05:
        ratio_name = "정사각형에 가까운 (5:4)"
    else:
        ratio_name = f"사용자 정의 ({ratio:.2f}:1)"
    
    # 파일 크기 예상 정보 추가
    megapixels = (w * h) / 1000000
    size_category = ""
    if megapixels < 1:
        size_category = "소형"
    elif megapixels < 5:
        size_category = "중형"
    elif megapixels < 20:
        size_category = "대형"
    else:
        size_category = "초대형"
    
    return f"📐 {w}×{h} | {ratio_name} | {megapixels:.1f}MP ({size_category})"


def show_image_info(image):
    """입력 이미지의 정보를 표시"""
    if image is None:
        return "이미지를 업로드해주세요."
    return get_aspect_ratio_info(image)


# ⭐ FluxKontext 모델용 스타일 프리셋 (인물 화질 최적화)
STYLE_PRESETS = {
    "기본": {
        "prompt_prefix": "ultra high quality, ultra detailed, 8k resolution, perfect skin, highly enhanced",
        "guidance_scale": 4.0,
        "num_inference_steps": 30,
    },
    "📸 인물 - 최고 화질": {
        "prompt_prefix": "ultra high quality portrait, 8k resolution, professional photography, perfect skin texture, natural lighting, detailed facial features, crystal clear, ultra detailed, masterpiece quality, professional grade",
        "guidance_scale": 4.8,
        "num_inference_steps": 40,
    },
    "👤 인물 - 자연스러운 보정": {
        "prompt_prefix": "natural portrait enhancement, soft lighting, realistic skin texture, subtle improvement, professional quality, high resolution, natural look",
        "guidance_scale": 4.2,
        "num_inference_steps": 35,
    },
    "🎭 인물 - 스튜디오 품질": {
        "prompt_prefix": "studio portrait, professional lighting, perfect skin, high-end photography, commercial quality, flawless details, premium grade",
        "guidance_scale": 5.0,
        "num_inference_steps": 45,
    },
    "🏞️ 풍경 - 선명도 향상": {
        "prompt_prefix": "stunning landscape, crystal clear details, vibrant colors, high dynamic range, professional landscape photography, ultra sharp, detailed environment",
        "guidance_scale": 4.2,
        "num_inference_steps": 32,
    },
    "🏔️ 풍경 - 자연색 복원": {
        "prompt_prefix": "natural landscape colors, realistic atmosphere, balanced exposure, detailed textures, professional nature photography, enhanced clarity",
        "guidance_scale": 3.8,
        "num_inference_steps": 30,
    },
    "📦 제품/사물 - 선명함": {
        "prompt_prefix": "product photography, studio quality, perfect lighting, sharp details, clean background, professional commercial photography, ultra clear",
        "guidance_scale": 5.2,
        "num_inference_steps": 35,
    },
    "🔍 제품/사물 - 질감 강화": {
        "prompt_prefix": "detailed texture enhancement, material definition, professional product shot, high resolution details, enhanced surface quality",
        "guidance_scale": 4.8,
        "num_inference_steps": 38,
    },
    "📄 문서/텍스트 - 가독성": {
        "prompt_prefix": "document enhancement, clear text, sharp typography, high contrast, readable content, professional document quality, clean scan",
        "guidance_scale": 6.0,
        "num_inference_steps": 28,
    },
    "📋 문서/텍스트 - 배경 정리": {
        "prompt_prefix": "clean document, white background, clear text, noise reduction, professional scan quality, enhanced readability",
        "guidance_scale": 5.5,
        "num_inference_steps": 30,
    },
    "🔧 이미지 복원 - 오래된 사진": {
        "prompt_prefix": "photo restoration, vintage photo enhancement, color correction, damage repair, restored quality, professional restoration",
        "guidance_scale": 4.8,
        "num_inference_steps": 45,
    },
    "✨ 이미지 복원 - 노이즈 제거": {
        "prompt_prefix": "noise reduction, image cleanup, quality enhancement, artifact removal, smooth restoration, professional grade",
        "guidance_scale": 4.5,
        "num_inference_steps": 38,
    },
}

# Gradio 인터페이스 생성
with gr.Blocks(
    title="FLUX.1-Kontext 이미지 향상기",
    theme=gr.themes.Soft(),
) as demo:
    
    # 상단 헤더
    with gr.Row():
        gr.Markdown(
            """
            # 🎨 FLUX.1-Kontext 이미지 향상기
            ## 🖼️ **Image-to-Image 전용**: 기존 이미지를 고품질로 향상시키거나 스타일을 변환합니다!
            ### ✅ **원본 크기 완전 유지**: 입력 이미지와 동일한 크기로 출력됩니다
            """
        )

    # 메인 이미지 영역 (좌우 정렬)
    with gr.Row(equal_height=True):
        # 왼쪽: 입력 이미지 + 컨트롤
        with gr.Column(scale=1):
            gr.Markdown("### 📸 입력 이미지")
            input_image = gr.Image(
                label="원본 이미지 업로드 (필수)",
                type="pil",
                sources=["upload", "clipboard"],
                height=400,
                #value="default.jpg",  # 기본 이미지 (예시용)
            )
            
            # ⭐ 입력 이미지 정보 표시
            image_info = gr.Textbox(
                label="📊 이미지 정보",
                value="이미지를 업로드해주세요.",
                interactive=False,
                lines=1
            )
            
            gr.Markdown("### 🎨 스타일 설정")
            style_preset = gr.Dropdown(
                label="스타일 프리셋",
                choices=list(STYLE_PRESETS.keys()),
                value="📸 인물 - 최고 화질",
                info="용도에 맞는 최적화된 설정이 자동으로 적용됩니다"
            )
            
            prompt_input = gr.Textbox(
                label="추가 요청사항 (선택사항)",
                placeholder="추가로 원하는 변환이나 개선사항을 입력하세요...",
                value="",
                lines=2
            )
            
            # 생성 버튼
            generate_btn = gr.Button(
                "🎨 이미지 향상 시작 (원본 크기 유지)", 
                variant="primary", 
                size="lg"
            )

        # 오른쪽: 출력 이미지 + 정보
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 향상된 이미지")
            output_image = gr.Image(
                label="향상된 이미지 (원본과 동일 크기)",
                type="pil",
                height=400
            )
            
            # 처리 정보
            info_output = gr.Textbox(
                label="📊 처리 정보", 
                lines=6, 
                interactive=False
            )

    # ⭐ 고급 설정 (프리셋에 따라 자동 조정됨)
    with gr.Accordion("⚙️ 고급 설정", open=False):
        with gr.Row():
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=4.8,  # 기본값: 인물 - 최고 화질 프리셋
                step=0.1,
                label="📐 가이던스 스케일",
                info="프롬프트 준수 정도 (프리셋에 따라 자동 설정됨)"
            )
            
            steps_slider = gr.Slider(
                minimum=15,
                maximum=50,
                value=40,   # 기본값: 인물 - 최고 화질 프리셋
                step=1,
                label="⚡ 추론 스텝 수",
                info="처리 단계 수 (프리셋에 따라 자동 설정됨)"
            )
            
        with gr.Row():
            max_sequence_length_slider = gr.Slider(
                minimum=128,
                maximum=512,
                value=320,
                step=32,
                label="📝 최대 시퀀스 길이",
                info="텍스트 프롬프트 처리 길이 (길수록 복잡한 프롬프트 처리 가능, 메모리 사용량 증가)"
            )
            
            seed_input = gr.Number(
                label="🎲 시드 (-1은 랜덤)",
                value=-1,
                precision=0,
                info="결과 재현용"
            )

    # 예제 프롬프트
    with gr.Accordion("📝 예제 프롬프트", open=False):
        gr.Examples(
            examples=[
                ["더 선명하고 깨끗하게"],
                ["자연스럽게 보정해주세요"],
                ["프로 사진작가 스타일로"],
                ["빈티지한 느낌으로"],
                ["따뜻한 조명으로"],
                ["흑백 예술 사진으로"],
            ],
            inputs=prompt_input
        )

    # 사용 가이드
    with gr.Accordion("💡 사용 가이드", open=False):
        gr.Markdown(
            """
            ### 🎯 원본 크기 완전 유지
            - **모든 이미지가 원본과 동일한 크기로 출력됩니다**
            - 1920x1080 입력 → 1920x1080 출력
            - 800x600 입력 → 800x600 출력
            - 어떤 크기든 완전히 동일하게 유지
            
            ### 👤 인물 사진 최적화
            - **📸 최고 화질**: 8K 전문가급 품질 (가이던스: 4.8, 스텝: 40)
            - **👤 자연스러운 보정**: 과하지 않은 개선 (가이던스: 4.2, 스텝: 35)
            - **🎭 스튜디오 품질**: 상업적 품질 (가이던스: 5.0, 스텝: 45)
            
            ### 💡 사용법
            1. 이미지 업로드 (어떤 크기든 가능)
            2. 스타일 프리셋 선택 (자동으로 최적 설정 적용)
            3. 필요 시 추가 요청사항 입력
            4. '이미지 향상 시작' 버튼 클릭
            5. 원본과 동일한 크기의 향상된 이미지 획득
            """
        )

    # ⭐ 이벤트 연결
    input_image.change(
        fn=show_image_info,
        inputs=[input_image],
        outputs=[image_info]
    )
    
    style_preset.change(
        fn=apply_preset_settings,
        inputs=[style_preset],
        outputs=[guidance_slider, steps_slider],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            input_image,
            guidance_slider,
            steps_slider,
            max_sequence_length_slider,  # Use the slider instead of gr.State(320)
            seed_input,
            style_preset,
        ],
        outputs=[output_image, info_output],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,
    )
