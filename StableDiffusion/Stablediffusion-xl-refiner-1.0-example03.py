import torch
import gradio as gr
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import datetime
import os

# 모델 로딩 (한 번만 실행)
print("모델 로딩 중...")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")
print("모델 로딩 완료!")


def resize_image_to_sdxl(image, max_size=1024):
    """이미지를 SDXL에 적합한 크기로 리사이즈하면서 비율 유지"""
    width, height = image.size
    aspect_ratio = width / height
    
    # 긴 쪽을 max_size로 맞추고 8의 배수로 조정
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)
    
    # 8의 배수로 조정 (Stable Diffusion 요구사항)
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    # 최소 크기 보장 (512x512)
    new_width = max(new_width, 512)
    new_height = max(new_height, 512)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def generate_image(input_image, prompt, negative_prompt, strength, guidance_scale, num_inference_steps, progress=gr.Progress()):
    if input_image is None:
        return None, "이미지를 업로드해주세요."
    
    if not prompt.strip():
        return None, "Prompt를 입력해주세요."
    
    try:
        progress(0, desc="이미지 처리 시작...")
        
        # PIL Image로 변환
        if isinstance(input_image, str):
            init_image = Image.open(input_image).convert("RGB")
        else:
            init_image = input_image.convert("RGB")
        
        # 원본 이미지 크기 정보
        original_width, original_height = init_image.size
        original_aspect_ratio = original_width / original_height
        
        progress(0.1, desc="이미지 크기 조정 중...")
        
        # 이미지를 SDXL에 적합한 크기로 리사이즈
        resized_image = resize_image_to_sdxl(init_image)
        new_width, new_height = resized_image.size
        
        progress(0.3, desc=f"이미지 생성 중... ({new_width}x{new_height})")
        
        # 이미지 생성
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=resized_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=new_width,
            height=new_height
        )
        
        progress(0.9, desc="이미지 저장 중...")
        
        generated_image = result.images[0]
        
        # 결과 이미지 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"sdxl_refined_{timestamp}.png"
        output_path = os.path.join(os.getcwd(), output_filename)
        generated_image.save(output_path)
        
        progress(1.0, desc="완료!")
        
        status_message = f"""이미지가 성공적으로 생성되었습니다!
저장 위치: {output_filename}
원본 크기: {original_width}x{original_height}
생성 크기: {new_width}x{new_height}
비율: {original_aspect_ratio:.2f}:1"""
        
        return generated_image, status_message
        
    except Exception as e:
        return None, f"오류가 발생했습니다: {str(e)}"


# Gradio 인터페이스 생성
with gr.Blocks(title="Stable Diffusion XL Refiner", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Stable Diffusion XL Refiner")
    gr.Markdown("입력 이미지를 업로드하고 프롬프트를 입력하여 고품질의 정제된 이미지를 생성하세요.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 입력 컨트롤
            input_image = gr.Image(
                label="입력 이미지",
                type="pil",
                height=300
            )
            
            prompt = gr.Textbox(
                label="Prompt (프롬프트)",
                placeholder="예: blue bikini, ultra high definition photo realistic portrait, similar to a photo",
                lines=3,
                value="ultra high definition photo realistic portrait, professional photography"
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt (네거티브 프롬프트)",
                placeholder="예: blurry, low quality, distorted, ugly",
                lines=2,
                value="blurry, low quality, distorted, ugly, deformed, bad anatomy"
            )
            
            with gr.Accordion("고급 설정", open=False):
                strength = gr.Slider(
                    label="Strength (변형 강도)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    info="높을수록 원본에서 더 많이 변형됩니다"
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
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=5,
                    info="생성 단계 수 (높을수록 품질 향상, 시간 증가)"
                )
            
            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            # 출력 결과
            output_image = gr.Image(
                label="생성된 이미지",
                height=400
            )
            
            status_text = gr.Textbox(
                label="상태",
                value="이미지를 업로드하고 '이미지 생성' 버튼을 클릭하세요.",
                interactive=False,
                lines=3
            )
    
    # 예시 이미지와 프롬프트
    gr.Markdown("## 📝 사용 팁")
    gr.Markdown("""
    - **이미지 비율**: 입력 이미지의 비율이 자동으로 유지되며, SDXL에 최적화된 크기로 조정됩니다
    - **권장 크기**: 최소 512x512, 최대 1024x1024 (긴 쪽 기준)
    - **Strength**: 0.3-0.5 (약간 수정), 0.6-0.8 (보통 수정), 0.9-1.0 (강한 수정)
    - **Guidance Scale**: 7-12 추천 (너무 높으면 과포화될 수 있음)
    - **좋은 프롬프트 예시**: "professional photography, high resolution, detailed, sharp focus"
    - **네거티브 프롬프트**: "blurry, low quality, distorted, ugly, deformed" 등으로 원하지 않는 요소 제거
    - **지원 비율**: 정사각형, 세로형, 가로형 모든 비율 지원 (8의 배수로 자동 조정)
    """)
    
    # 이벤트 연결
    generate_btn.click(
        fn=generate_image,
        inputs=[input_image, prompt, negative_prompt, strength, guidance_scale, num_inference_steps],
        outputs=[output_image, status_text],
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
