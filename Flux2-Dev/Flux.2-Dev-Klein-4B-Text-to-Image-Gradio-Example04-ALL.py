import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import gradio as gr

DEFAULT_PROMPT = "4k, A highly realistic, high-quality photo of a beautiful Instagram-style cute korean girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles, wearing a cute red bikini. The image should capture her in a half-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

# Global variables for model
pipe = None
demo = None

def get_device_and_dtype():
    """Detect the best available device and appropriate dtype."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"CUDA 감지됨: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
        print("Apple MPS 감지됨")
    else:
        device = "cpu"
        dtype = torch.float32  # CPU에서는 float32가 더 안정적
        print("CPU 모드로 실행")
    return device, dtype

DEVICE, DTYPE = get_device_and_dtype()

def cleanup():
    """Release all resources safely."""
    global pipe, demo
    print("\n리소스 정리 중...")

    # Close Gradio demo
    if demo is not None:
        try:
            demo.close()
            print("Gradio 인터페이스 종료됨")
        except Exception as e:
            print(f"Gradio 종료 중 오류: {e}")

    # Release pipeline and GPU memory
    if pipe is not None:
        try:
            del pipe
            pipe = None
            print("모델 메모리 해제됨")
        except Exception as e:
            print(f"모델 해제 중 오류: {e}")

    # Clear GPU/MPS cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA 캐시 정리됨")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing, but gc helps
        pass

    gc.collect()
    print("리소스 정리 완료!")

def signal_handler(_signum, _frame):
    """Handle keyboard interrupt signal."""
    print("\n\nKeyboard interrupt 감지됨 (Ctrl+C)")
    cleanup()
    sys.exit(0)

def load_model():
    """Load and initialize the Flux2Klein model with optimizations."""
    global pipe

    print(f"모델 로딩 중... (Device: {DEVICE}, Dtype: {DTYPE})")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=DTYPE
    )

    pipe = pipe.to(DEVICE)

    print("모델 로딩 완료!")
    return pipe

def generate_image(prompt, height, width, guidance_scale, strength, num_inference_steps, seed):
    """Generate image from text prompt and return for UI display."""
    global pipe
    
    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."
    
    try:
        print(f"출력 크기: {width}x{height}")

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

        # Strength is only passed if the pipeline supports it to avoid runtime errors.
        if hasattr(pipe, "__call__") and "strength" in pipe.__call__.__code__.co_varnames:
            pipe_kwargs["strength"] = strength

        print(f"이미지 생성 중... (steps: {num_inference_steps}, strength: {strength})")
        image = pipe(**pipe_kwargs).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}.png"
        image.save(output_path)
        print(f"이미지 저장됨: {output_path}")
        
        return image, f"✓ 이미지 생성 완료! 저장됨: {output_path}"
    
    except Exception as e:
        return None, f"오류: {str(e)}"

def main():
    global demo

    # Register signal handler for keyboard interrupt
    signal.signal(signal.SIGINT, signal_handler)

    # Load model once at startup
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.2 Dev Klein 4B 이미지 생성기") as demo:
        gr.Markdown("# Flux.2 Dev Klein 4B Text-to-Image 생성기")
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
                            value=512
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
                            maximum=50,
                            step=1,
                            value=4
                        )
                        strength_input = gr.Slider(
                            label="Strength",
                            info="노이즈 비율 (0: 원본 유지, 1: 완전한 노이즈)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.5
                        )
                    
                    seed_input = gr.Slider(
                        label="시드 (Seed)",
                        info="재현성을 위한 난수 시드 (0: 랜덤)",
                        minimum=-1,
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
                strength_input,
                steps_input,
                seed_input
            ],
            outputs=[image_output, status_output]
        )
    
    # Launch the interface
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt 감지됨 (Ctrl+C)")
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n예상치 못한 오류 발생: {e}")
        cleanup()
        sys.exit(1)
