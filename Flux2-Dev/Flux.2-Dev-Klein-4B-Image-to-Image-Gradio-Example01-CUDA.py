import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gradio as gr

DEFAULT_PROMPT = "Highly realistic, 4k, high-quality, high resolution, beautiful skinny korean woman model walking on a sunny beach."

DEFAULT_IMAGE = None  # Will be set to default.png or default.jpg if available

# Global variables for model
DEVICE = "cuda"
DTYPE = torch.bfloat16
pipe = None

def load_model():
    """Load and initialize the Flux2Klein model with optimizations."""
    global pipe, DEFAULT_IMAGE
    
    print("모델 로딩 중...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=DTYPE
    )
    pipe = pipe.to(DEVICE)
    
    # Memory optimization 
    pipe.enable_model_cpu_offload() # CUDA에서 CPU RAM을 일부 사용
    pipe.enable_attention_slicing() # 안쓰면 GPU 메모리를 더 사용함(속)
    pipe.enable_sequential_cpu_offload() # 안쓰면 CUDA에서 느림
    
    # Load default image if available
    if os.path.exists("default.png"):
        DEFAULT_IMAGE = Image.open("default.png")
        print("default.png 로딩 완료")
    elif os.path.exists("default.jpg"):
        DEFAULT_IMAGE = Image.open("default.jpg")
        print("default.jpg 로딩 완료")
    else:
        print("기본 이미지가 없습니다. default.png 또는 default.jpg를 추가하세요.")
    
    print("모델 로딩 완료!")
    return pipe

def update_image_dimensions(input_image):
    """입력 이미지의 크기를 반환하여 슬라이더 업데이트"""
    if input_image is None:
        return 1024, 512
    
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    
    width, height = input_image.size
    return height, width

def generate_image(input_image, prompt, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image from input image and text prompt."""
    global pipe
    
    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."
    
    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)
        
        # Resize input image to match output dimensions
        input_image = input_image.resize((width, height))
        print(f"출력 크기: {width}x{height}")

        # Setup generator
        generator = torch.Generator(device=DEVICE)
        if seed is not None and seed >= 0:
            generator.manual_seed(seed)

        # Generate image with image-to-image
        pipe_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        print(f"이미지 편집 중... (steps: {num_inference_steps})")
        image = pipe(**pipe_kwargs).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}.png"
        image.save(output_path)
        print(f"이미지 편집 완료! 저장됨: {output_path}")
        
        return image, f"✓ 이미지 편집 완료! 저장됨: {output_path}"
    
    except Exception as e:
        return None, f"오류: {str(e)}"

def main():
    # Load model once at startup
    load_model()
    
    # Create Gradio interface
    with gr.Blocks(title="Flux.2 Klein 4B Image-to-Image 편집기") as demo:
        gr.Markdown("# Flux.2 Klein 4B Image-to-Image 편집기")
        gr.Markdown("입력 이미지와 텍스트 설명을 입력하여 이미지를 편집하세요.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="입력 이미지",
                    type="pil",
                    value=DEFAULT_IMAGE,
                    height=400
                )
                prompt_input = gr.Textbox(
                    label="이미지 설명 (영어)",
                    placeholder="예: transform this into a cyberpunk style",
                    value=DEFAULT_PROMPT,
                    lines=5
                )
                
                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        height_input = gr.Slider(
                            label="높이 (Height)",
                            info="생성할 이미지의 높이 (픽셀, 입력 이미지 로드 시 자동 설정)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                        width_input = gr.Slider(
                            label="너비 (Width)",
                            info="생성할 이미지의 너비 (픽셀, 입력 이미지 로드 시 자동 설정)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=512
                        )
                    
                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=1.0,
                            maximum=5.0,
                            step=0.5,
                            value=1.0
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=4
                        )
                    
                    seed_input = gr.Slider(
                        label="시드 (Seed)",
                        info="재현성을 위한 난수 시드 (0: 랜덤)",
                        minimum=-1,
                        maximum=1000,
                        step=1,
                        value=42
                    )
                
                submit_btn = gr.Button("이미지 편집", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(label="출력 이미지", height=800)
                status_output = gr.Textbox(label="상태", interactive=False)
        
        # 입력 이미지가 변경되면 자동으로 크기를 슬라이더에 반영
        image_input.change(
            fn=update_image_dimensions,
            inputs=[image_input],
            outputs=[height_input, width_input]
        )
        
        # Connect button to generation function
        submit_btn.click(
            fn=generate_image,
            inputs=[
                image_input,
                prompt_input,
                height_input,
                width_input,
                guidance_input,
                steps_input,
                seed_input
            ],
            outputs=[image_output, status_output]
        )
    
    # Launch the interface
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()
