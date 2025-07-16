import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image


# 모델 로딩 (한 번만 실행)
try:
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.enable_model_cpu_offload()
    base = base.to("cpu")
    print("> stable-diffusion-xl-base-1.0 모델 로드 성공")
except Exception as e:
    print(f"베이스 모델 로딩 중 오류가 발생했습니다: {e}")
    base = None

try: 
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        )
    refiner.enable_model_cpu_offload()
    refiner = refiner.to("cpu")
    print("> stable-diffusion-xl-refiner-1.0 모델 로드 성공")
except Exception as e:
    print(f"리파이너 모델 로딩 중 오류가 발생했습니다: {e}")
    refiner = None


def resize_image_to_sdxl(image, max_size=1024):
    """이미지를 SDXL에 적합한 크기로 리사이즈하면서 원본 크기 최대한 유지 (현재는 사용하지 않음)"""
    width, height = image.size
    
    # 원본 크기가 너무 큰 경우에만 다운스케일
    if max(width, height) > max_size:
        aspect_ratio = width / height
        if width > height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)
    else:
        # 원본 크기 유지
        new_width = width
        new_height = height
    
    # 8의 배수로 조정 (Stable Diffusion 요구사항)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # 최소 크기 보장 (512px 미만인 경우만)
    if new_width < 512:
        new_width = 512
    if new_height < 512:
        new_height = 512
    
    # 원본과 크기가 같으면 리사이즈 하지 않음
    if new_width == width and new_height == height:
        return image
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def generate_image(prompt, negative_prompt, width, height, strength, guidance_scale, num_inference_steps, progress=gr.Progress()):
    if not prompt.strip():
        return None, None, "Prompt를 입력해주세요."
    
    if base is None or refiner is None:
        return None, None, "모델이 로드되지 않았습니다."
    
    try:
        progress(0, desc="이미지 생성 시작...")
        
        # 크기를 8의 배수로 조정
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        progress(0.1, desc="베이스 이미지 생성 중...")
        
        # 1단계: 베이스 모델로 이미지 생성
        base_image = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            denoising_end=1.0,  # 베이스는 ?(소수점)까지만 생성
            output_type="latent"  # latent 형태로 출력하여 refiner에 전달
        ).images[0]
        
        progress(0.6, desc="리파이너로 이미지 정제 중...")
        
        # 2단계: 리파이너로 이미지 정제
        refined_image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            num_inference_steps=num_inference_steps,
            denoising_start=0.8,  # 베이스에서 80% 완료된 지점부터 시작
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
        
        progress(0.9, desc="이미지 저장 중...")
        
        # 베이스 이미지도 PIL로 변환해서 비교용으로 저장
        base_pil = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            denoising_end=0.8
        ).images[0]
        
        progress(1.0, desc="완료!")
        status_message = "이미지가 성공적으로 생성되었습니다!"
        return base_pil, refined_image, status_message
        
    except Exception as e:
        return None, None, f"오류가 발생했습니다: {str(e)}"


# Gradio 인터페이스 생성
with gr.Blocks(title="Stable Diffusion XL Base + Refiner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Stable Diffusion XL Base + Refiner")
    gr.Markdown("프롬프트를 입력하여 베이스 모델로 이미지를 생성하고, 리파이너로 고품질 이미지를 만들어보세요.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 입력 컨트롤
            prompt = gr.Textbox(
                label="Prompt (프롬프트)",
                placeholder="예: a beautiful woman in red bikini walking on sunny beach",
                lines=3,
                value="a beautiful woman in red bikini walking on sunny beach, ultra high quality, ultra detail, photorealistic, vibrant colors, full body, good fingers"
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt (네거티브 프롬프트)",
                placeholder="예: blurry, low quality, distorted, ugly",
                lines=2,
                value="blurry, low quality, distorted, ugly, deformed, bad anatomy"
            )
            
            with gr.Row():
                width = gr.Slider(
                    label="너비",
                    minimum=512,
                    maximum=1536,
                    value=1024,
                    step=64,
                    info="8의 배수로 자동 조정됩니다"
                )
                
                height = gr.Slider(
                    label="높이",
                    minimum=512,
                    maximum=1536,
                    value=1024,
                    step=64,
                    info="8의 배수로 자동 조정됩니다"
                )
            
            with gr.Accordion("고급 설정", open=False):
                strength = gr.Slider(
                    label="Refiner Strength (리파이너 강도)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    info="리파이너의 변형 강도"
                )
                
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.5,
                    info="프롬프트 준수 정도"
                )
                
                num_inference_steps = gr.Slider(
                    label="Inference Steps",
                    minimum=20,
                    maximum=50,
                    value=40,
                    step=5,
                    info="생성 단계 수 (높을수록 품질 향상, 시간 증가)"
                )
            
            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")
            
        with gr.Column(scale=2):
            # 출력 결과
            with gr.Row():
                base_output = gr.Image(
                    label="베이스 이미지 (Base Model)",
                    height=350
                )
                
                refined_output = gr.Image(
                    label="리파인된 이미지 (Refined)",
                    height=350
                )
            
            status_text = gr.Textbox(
                label="상태",
                value="프롬프트를 입력하고 '이미지 생성' 버튼을 클릭하세요.",
                interactive=False,
                lines=4
            )
    
    # 예시 이미지와 프롬프트
    gr.Markdown("## 📝 사용 팁")
    gr.Markdown("""
    - **2단계 생성**: 베이스 모델로 초기 이미지 생성 → 리파이너로 품질 향상
    - **denoising 분할**: 베이스 80% → 리파이너 20%로 효율적 처리
    - **해상도**: 512px~1536px 지원, 8의 배수로 자동 조정
    - **Refiner Strength**: 0.1-0.3 (약간 개선), 0.4-0.7 (보통 개선), 0.8-1.0 (강한 개선)
    - **Guidance Scale**: 7-12 추천 (너무 높으면 과포화될 수 있음)
    - **Inference Steps**: 30-50 추천 (높을수록 고품질, 시간 증가)
    - **좋은 프롬프트 예시**: "photorealistic portrait, professional photography, high quality, detailed"
    - **네거티브 프롬프트**: "blurry, low quality, distorted, ugly, deformed" 등으로 원하지 않는 요소 제거
    """)
    
    # 이벤트 연결
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, width, height, strength, guidance_scale, num_inference_steps],
        outputs=[base_output, refined_output, status_text],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
