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

if torch.cuda.is_available():
    pipe = pipe.to("cuda")
else:
    print("CUDA가 사용 불가능합니다!!!");
    exit(1)
    
print("모델 로딩 완료!")


def resize_image_to_sdxl(image):
    """이미지의 비율은 유지하면서, 가로/세로가 16의 배수로만 맞춤"""
    width, height = image.size

    def round_to_16(x):
        return max(16, (x // 16) * 16)

    new_width = round_to_16(width)
    new_height = round_to_16(height)

    # 크기가 변경되면 리사이즈, 아니면 원본 반환
    if new_width != width or new_height != height:
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


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
        
        # 크기 변경 여부 확인
        size_changed = (new_width != original_width or new_height != original_height)
        size_info = f"크기 유지: {new_width}x{new_height}" if not size_changed else f"크기 조정: {original_width}x{original_height} → {new_width}x{new_height}"
        
        progress(0.3, desc=f"이미지 생성 중... ({size_info})")
        
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
        
        # 크기 변경 여부에 따른 메시지
        size_change_msg = "크기 유지됨" if not size_changed else f"크기 조정됨 ({original_width}x{original_height} → {new_width}x{new_height})"
        
        status_message = f"""이미지가 성공적으로 생성되었습니다!
저장 위치: {output_filename}
원본 크기: {original_width}x{original_height}
생성 크기: {new_width}x{new_height}
처리 결과: {size_change_msg}
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
                height=500
            )
            
            prompt = gr.Textbox(
                label="Prompt (프롬프트)",
                placeholder="예: blue bikini, ultra high definition photo realistic portrait, similar to a photo",
                lines=3,
                value="ultra high definition photo realistic portrait, professional photography, ultra detail, similar to a photo, no deformed"
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
    - **이미지 크기**: 원본 크기를 최대한 유지하며, 1024px를 초과하는 경우에만 다운스케일됩니다
    - **자동 조정**: 8의 배수로 자동 조정되어 SDXL과 완벽 호환됩니다
    - **최소 크기**: 512px 미만인 경우 512px로 업스케일됩니다
    - **Strength**: 0.3-0.5 (약간 수정), 0.6-0.8 (보통 수정), 0.9-1.0 (강한 수정)
    - **Guidance Scale**: 7-12 추천 (너무 높으면 과포화될 수 있음)
    - **좋은 프롬프트 예시**: "professional photography, high resolution, detailed, sharp focus"
    - **네거티브 프롬프트**: "blurry, low quality, distorted, ugly, deformed" 등으로 원하지 않는 요소 제거
    - **지원 해상도**: 모든 비율과 크기 지원 (원본에 최대한 가깝게 유지)
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
