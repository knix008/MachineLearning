import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os

# 전역 변수로 모델 저장 (한 번만 로드)
pipe = None

def load_model_once():
    """모델을 한 번만 로드하여 재사용합니다."""
    global pipe
    if pipe is None:
        print("모델을 로드하는 중...")
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            print("GPU 모드로 실행 중")
        else:
            print("CPU 모드로 실행 중")
    
    return pipe

def quick_style_transfer(image, style_prompt, strength=0.6):
    """빠른 스타일 변환을 수행합니다."""
    try:
        # 모델 로드
        model = load_model_once()
        
        # 이미지 전처리
        if image is None:
            return None, "이미지를 업로드해주세요."
        
        # 이미지 크기 조정 (빠른 처리를 위해)
        max_size = 512
        width, height = image.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # 스타일 변환
        result = model(
            prompt=style_prompt,
            image=image,
            strength=strength,
            guidance_scale=7.0,
            num_inference_steps=15
        )
        
        return result.images[0], "변환 완료!"
        
    except Exception as e:
        return None, f"오류: {str(e)}"

def create_simple_interface():
    """간단한 Gradio 인터페이스를 생성합니다."""
    
    # 미리 정의된 스타일들
    preset_styles = {
        "유화화": "oil painting, masterpiece, detailed, vibrant colors",
        "수채화": "watercolor painting, soft, dreamy, artistic",
        "애니메": "anime style, cel shading, vibrant, detailed",
        "스케치": "sketch, pencil drawing, black and white, artistic",
        "인상주의": "impressionist painting, brush strokes, colorful",
        "사이버펑크": "cyberpunk style, neon lights, futuristic",
        "빈티지": "vintage, retro, 1950s style, nostalgic",
        "판타지": "fantasy art, magical, ethereal, mystical"
    }
    
    with gr.Blocks(title="간단한 스타일 변환", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎨 간단한 이미지 스타일 변환")
        gr.Markdown("이미지를 업로드하고 스타일을 선택하세요!")
        
        with gr.Row():
            with gr.Column():
                # 입력
                input_image = gr.Image(
                    label="변환할 이미지",
                    type="pil",
                    height=300
                )
                
                # 스타일 선택
                style_dropdown = gr.Dropdown(
                    choices=list(preset_styles.keys()),
                    label="스타일 선택",
                    value="유화화"
                )
                
                # 커스텀 프롬프트
                custom_prompt = gr.Textbox(
                    label="커스텀 스타일 (선택사항)",
                    placeholder="원하는 스타일을 직접 입력하세요",
                    lines=2
                )
                
                # 변환 강도
                strength = gr.Slider(
                    minimum=0.3,
                    maximum=0.8,
                    value=0.6,
                    step=0.1,
                    label="변환 강도",
                    info="높을수록 더 많이 변환됩니다"
                )
                
                # 변환 버튼
                transform_btn = gr.Button(
                    "🎨 변환하기",
                    variant="primary",
                    size="lg"
                )
                
                # 상태
                status = gr.Textbox(
                    label="상태",
                    interactive=False
                )
            
            with gr.Column():
                # 출력
                output_image = gr.Image(
                    label="변환된 이미지",
                    height=400
                )
        
        # 스타일 버튼들
        gr.Markdown("## 💡 빠른 스타일 선택")
        with gr.Row():
            for style_name in list(preset_styles.keys())[:4]:
                gr.Button(
                    style_name,
                    size="sm"
                ).click(
                    fn=lambda x=style_name: x,
                    outputs=style_dropdown
                )
        
        with gr.Row():
            for style_name in list(preset_styles.keys())[4:]:
                gr.Button(
                    style_name,
                    size="sm"
                ).click(
                    fn=lambda x=style_name: x,
                    outputs=style_dropdown
                )
        
        # 변환 함수
        def process_image(image, selected_style, custom_prompt, strength):
            if custom_prompt.strip():
                prompt = custom_prompt
            else:
                prompt = preset_styles[selected_style]
            
            return quick_style_transfer(image, prompt, strength)
        
        # 이벤트 연결
        transform_btn.click(
            fn=process_image,
            inputs=[input_image, style_dropdown, custom_prompt, strength],
            outputs=[output_image, status]
        )
        
        # 사용법
        gr.Markdown("""
        ## 📖 사용법
        
        1. **이미지 업로드**: 변환할 이미지를 드래그 앤 드롭하거나 클릭하여 업로드
        2. **스타일 선택**: 미리 정의된 스타일 중 선택하거나 커스텀 스타일 입력
        3. **강도 조정**: 변환 강도를 0.3-0.8 사이에서 조정
        4. **변환 실행**: "변환하기" 버튼 클릭
        
        ### 💡 팁
        - **빠른 처리**: 작은 이미지(512px 이하)가 더 빠릅니다
        - **품질**: 변환 강도 0.6 정도가 좋은 균형점입니다
        - **커스텀**: 원하는 스타일을 자유롭게 입력할 수 있습니다
        """)
    
    return interface

if __name__ == "__main__":
    # 인터페이스 실행
    interface = create_simple_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True
    ) 