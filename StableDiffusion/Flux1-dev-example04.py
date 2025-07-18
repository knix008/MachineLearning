import torch
import gradio as gr
import time
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
import os

# Load model with memory optimizations
print("모델을 로딩 중입니다...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
pipe.enable_vae_slicing()  # Slice VAE computation
print("모델 로딩 완료!")

# 기본 이미지 로드 함수
def load_default_image():
    """기본 이미지 파일이 존재하면 로드"""
    default_path = "cloe-test01.jpg"
    if os.path.exists(default_path):
        try:
            return Image.open(default_path)
        except Exception as e:
            print(f"기본 이미지 로드 실패: {e}")
            return None
    return None

def generate_image(prompt, input_image, width, height, guidance_scale, num_inference_steps, max_sequence_length, strength, seed):
    """이미지 생성 함수 (텍스트-투-이미지 또는 이미지-투-이미지)"""
    start_time = time.time()
    
    # 폭과 높이를 16으로 나누어지도록 조정 (축소 방향으로만, 비율 유지)
    width = int(width)
    height = int(height)
    
    # 원본 비율 계산
    aspect_ratio = width / height
    
    # 16의 배수로 내림 (축소)
    adjusted_width = (width // 16) * 16
    adjusted_height = (height // 16) * 16
    
    # 최소 크기 512x512 보장하면서 비율 유지
    if adjusted_width < 512 or adjusted_height < 512:
        if aspect_ratio >= 1.0:  # 가로가 더 크거나 같은 경우
            adjusted_height = 512
            adjusted_width = int(512 * aspect_ratio)
            # 16의 배수로 조정
            adjusted_width = (adjusted_width // 16) * 16
        else:  # 세로가 더 큰 경우
            adjusted_width = 512
            adjusted_height = int(512 / aspect_ratio)
            # 16의 배수로 조정
            adjusted_height = (adjusted_height // 16) * 16
    
    # 최종 최소 크기 확인
    adjusted_width = max(adjusted_width, 512)
    adjusted_height = max(adjusted_height, 512)
    
    # 시드 설정
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator("cpu").manual_seed(seed)
    
    try:
        # 입력 이미지가 있는 경우 img2img, 없는 경우 txt2img
        if input_image is not None:
            # 입력 이미지 크기 조정
            input_image = input_image.resize((adjusted_width, adjusted_height), Image.LANCZOS)
            
            # img2img 생성 - 원본 이미지 보존을 위한 최적화된 설정
            image = pipe(
                prompt,
                image=input_image,
                height=adjusted_height,
                width=adjusted_width,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                max_sequence_length=int(max_sequence_length),
                strength=min(strength, 0.8),  # 최대 0.8로 제한하여 원본 더 보존
                generator=generator
            ).images[0]
            
            generation_type = "이미지-투-이미지"
        else:
            # 기본 이미지가 있는지 확인
            default_image = load_default_image()
            if default_image is not None:
                # 기본 이미지를 사용한 img2img
                default_image = default_image.resize((adjusted_width, adjusted_height), Image.LANCZOS)
                
                image = pipe(
                    prompt,
                    image=default_image,
                    height=adjusted_height,
                    width=adjusted_width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=int(num_inference_steps),
                    max_sequence_length=int(max_sequence_length),
                    strength=0.5,  # 기본 이미지 사용 시 낮은 strength로 원본 특성 유지
                    generator=generator
                ).images[0]
                
                generation_type = "기본이미지-투-이미지"
            else:
                # txt2img 생성
                image = pipe(
                    prompt,
                    height=adjusted_height,
                    width=adjusted_width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=int(num_inference_steps),
                    max_sequence_length=int(max_sequence_length),
                    generator=generator
                ).images[0]
                
                generation_type = "텍스트-투-이미지"
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # 이미지 저장
        timestamp = int(time.time())
        filename = f"flux_generated_{timestamp}.png"
        image.save(filename)
        
        # 크기 조정 정보 포함
        size_info = ""
        if width != adjusted_width or height != adjusted_height:
            original_ratio = width / height
            final_ratio = adjusted_width / adjusted_height
            size_info = f"\n크기 조정: {width}x{height} → {adjusted_width}x{adjusted_height}"
            size_info += f"\n비율 유지: {original_ratio:.2f} → {final_ratio:.2f}"
        
        info_text = f"생성 완료! ({generation_type})\n시간: {generation_time:.2f}초\n시드: {seed}\n저장된 파일: {filename}{size_info}"
        
        return image, info_text
        
    except Exception as e:
        error_text = f"오류 발생: {str(e)}"
        return None, error_text

def update_ui_visibility(input_image):
    """입력 이미지에 따라 UI 요소 표시/숨김"""
    if input_image is not None:
        return gr.update(visible=True), gr.update(value="이미지를 프롬프트에 맞게 수정합니다...")
    else:
        return gr.update(visible=False), gr.update(value="생성하고 싶은 이미지를 설명해주세요...")

# Gradio 인터페이스 생성
with gr.Blocks(title="FLUX.1-dev 이미지 생성기") as demo:
    gr.Markdown("# 🎨 FLUX.1-dev 이미지 생성기")
    gr.Markdown("텍스트로 새 이미지를 생성하거나, 기존 이미지를 프롬프트에 맞게 수정하세요!")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 입력 이미지 (선택사항)
            input_image = gr.Image(
                label="입력 이미지 (선택사항)",
                type="pil",
                sources=["upload", "clipboard"],
                value=load_default_image()
            )
            
            # 입력 컨트롤들
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지를 설명해주세요...",
                value="full body, good hands, good hair, good legs, skinny, blue eyes, photorealistic, 8k resolution, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, looking at viewer, perfect anatomy",
                lines=4
            )
            
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64,
                    label="너비"
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=768,
                    step=64,
                    label="높이"
                )
            
            guidance_slider = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=3.5,
                step=0.1,
                label="가이던스 스케일"
            )
            
            steps_slider = gr.Slider(
                minimum=10,
                maximum=50,
                value=28,
                step=1,
                label="추론 스텝 수"
            )
            
            sequence_slider = gr.Slider(
                minimum=128,
                maximum=512,
                value=256,
                step=32,
                label="최대 시퀀스 길이"
            )
            
            # 이미지-투-이미지 전용 설정
            strength_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,  # 기본값을 0.5로 낮춤 (원본 더 보존)
                step=0.1,
                label="변형 강도 (낮을수록 원본 유지)",
                visible=False
            )
            
            seed_input = gr.Number(
                label="시드 (-1은 랜덤)",
                value=-1,
                precision=0
            )
            
            generate_btn = gr.Button("🎨 이미지 생성", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # 출력 영역
            output_image = gr.Image(
                label="생성된 이미지",
                type="pil",
                height=500
            )
            
            info_output = gr.Textbox(
                label="생성 정보",
                lines=4,
                interactive=False
            )
    
    # 이벤트 연결
    input_image.change(
        fn=update_ui_visibility,
        inputs=[input_image],
        outputs=[strength_slider, prompt_input]
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            input_image,
            width_slider,
            height_slider,
            guidance_slider,
            steps_slider,
            sequence_slider,
            strength_slider,
            seed_input
        ],
        outputs=[output_image, info_output]
    )
    
    # 예제 프롬프트
    gr.Examples(
        examples=[
            ["a cute cat holding a sign that says hello world"],
            ["a futuristic city skyline at sunset, cyberpunk style"],
            ["a beautiful landscape with mountains and a lake, oil painting style"],
            ["a portrait of a woman with blue eyes, renaissance painting style"],
            ["a magical forest with glowing mushrooms, fantasy art"],
            ["convert this image to anime style, vibrant colors"],
            ["make this image look like a watercolor painting"],
            ["transform this to a cyberpunk style with neon lights"]
        ],
        inputs=prompt_input
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
