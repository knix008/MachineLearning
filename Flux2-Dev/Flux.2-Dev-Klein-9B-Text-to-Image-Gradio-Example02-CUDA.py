import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gradio as gr

DEFAULT_PROMPT = "a glamorous hot skinny korean girl, wearing a red bikini, posing on a tropical beach at sunset, cinematic lighting, 4k, ultra-detailed texture, with perfect anatomy, beautiful body, fashion vibe."

# Global variables for model
DEVICE = "cuda"
DTYPE = torch.bfloat16
pipe = None

def load_model():
    """Load and initialize the Flux2Klein model with optimizations."""
    global pipe
    
    print("모델 로딩 중...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=DTYPE
    )
    pipe = pipe.to(DEVICE)
    
    # Memory optimization 
    pipe.enable_model_cpu_offload() # CUDA에서 CPU RAM을 일부 사용
    pipe.enable_attention_slicing() # 안쓰면 GPU 메모리를 더 사용함(속)
    pipe.enable_sequential_cpu_offload() # 안쓰면 CUDA에서 더 빠름(4 추론스텝에서 1초 단축), CPU에서는 사용해야 함
    
    print("모델 로딩 완료!")
    return pipe

def generate_image(prompt, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image from text prompt and return for UI display."""
    global pipe
    
    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."
    
    try:
        print(f"출력 크기: {width}x{height}")
        print(f"추론 스텝: {num_inference_steps}, 시드: {int(seed)}")

        # Setup generator
        generator = torch.Generator(device=DEVICE)
        if seed is not None and seed >= 0:
            generator.manual_seed(seed)

        # Generate image
        pipe_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }

        print(f"이미지 생성 중... (steps: {num_inference_steps}, seed: {int(seed)})")
        image = pipe(**pipe_kwargs).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}_step{num_inference_steps}_seed{int(seed)}.png"
        image.save(output_path)
        print(f"이미지 저장됨: {output_path}")
        
        return image, f"✓ 이미지 생성 완료! 저장됨: {output_path}"
    
    except Exception as e:
        return None, f"오류: {str(e)}"

def main():
    # Load model once at startup
    load_model()
    
    # Create Gradio interface
    with gr.Blocks(title="Flux.2 Dev Klein 9B 이미지 생성기") as demo:
        gr.Markdown("# Flux.2 Dev Klein 9B Text-to-Image 생성기")
        gr.Markdown("텍스트 설명을 입력하여 이미지를 생성하세요.")
        
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="이미지 설명 (영어)",
                    placeholder="예: a beautiful landscape with mountains and a lake",
                    value=DEFAULT_PROMPT,
                    lines=5
                )
                
                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        height_input = gr.Slider(
                            label="높이 (Height)",
                            info="생성할 이미지의 높이 (픽셀)",
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=1024
                        )
                        width_input = gr.Slider(
                            label="너비 (Width)",
                            info="생성할 이미지의 너비 (픽셀)",
                            minimum=256,
                            maximum=1024,
                            step=64,
                            value=768
                        )
                    
                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=4,
                            maximum=20,
                            step=1,
                            value=4
                        )
                    
                    seed_input = gr.Slider(
                        label="시드 (Seed)",
                        info="재현성을 위한 난수 시드 (0: 랜덤)",
                        minimum=0,
                        maximum=1000,
                        step=1,
                        value=42    
                    )
                
                submit_btn = gr.Button("이미지 생성", variant="primary", size="lg")
            
            with gr.Column():
                image_output = gr.Image(label="출력 이미지", height=800)
                status_output = gr.Textbox(label="상태", interactive=False)
        
        # Connect button to generation function
        submit_btn.click(
            fn=generate_image,
            inputs=[
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
