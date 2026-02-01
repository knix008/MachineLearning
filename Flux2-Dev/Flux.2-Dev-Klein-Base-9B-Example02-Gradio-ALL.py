import torch
import platform
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os
import gc
import signal
import sys
import psutil
import gradio as gr

DEFAULT_PROMPT = "A highly realistic, high-quality photo of a beautiful Instagram-style korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

#prompt = "Highly realistic, 4k, high-quality, high resolution, beautiful full body korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw."


def print_hardware_info():
    """Print detailed hardware specifications."""
    print("=" * 60)
    print("하드웨어 사양")
    print("=" * 60)

    # OS 정보
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OS 버전: {platform.version()}")
    print(f"아키텍처: {platform.machine()}")

    # Python 정보
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # CPU 정보
    print("-" * 60)
    print("CPU 정보")
    print("-" * 60)
    print(f"프로세서: {platform.processor()}")
    print(f"물리 코어: {psutil.cpu_count(logical=False)}")
    print(f"논리 코어: {psutil.cpu_count(logical=True)}")

    # 메모리 정보
    mem = psutil.virtual_memory()
    print("-" * 60)
    print("메모리 정보")
    print("-" * 60)
    print(f"총 RAM: {mem.total / (1024**3):.1f} GB")
    print(f"사용 가능: {mem.available / (1024**3):.1f} GB")
    print(f"사용률: {mem.percent}%")

    # GPU 정보
    print("-" * 60)
    print("GPU 정보")
    print("-" * 60)
    if torch.cuda.is_available():
        print("CUDA 사용 가능: 예")
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - VRAM: {props.total_memory / (1024**3):.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - 멀티프로세서: {props.multi_processor_count}")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) 사용 가능: 예")
        print(f"디바이스: {platform.processor()}")
    else:
        print("GPU 사용 불가 (CPU 모드)")

    print("=" * 60)


def get_device_and_dtype():
    """Detect the best available device and appropriate data type."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    return device, dtype


# Global variables for model
DEVICE, DTYPE = get_device_and_dtype()
pipe = None
demo = None

def cleanup():
    """Release all resources and clear memory."""
    global pipe, demo
    print("\n리소스 정리 중...")

    if demo is not None:
        try:
            demo.close()
            print("Gradio 서버 종료됨")
        except Exception:
            pass
        demo = None

    if pipe is not None:
        del pipe
        pipe = None
        print("모델 해제됨")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA 캐시 정리됨")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("MPS 캐시 정리됨")

    gc.collect()
    print("메모리 정리 완료")

def signal_handler(_sig, _frame):
    """Handle keyboard interrupt signal."""
    print("\n\n키보드 인터럽트 감지됨 (Ctrl+C)")
    cleanup()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def load_model():
    """Load and initialize the Flux2Klein Base model with optimizations."""
    global pipe

    print("모델 로딩 중...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-9B",
        torch_dtype=DTYPE
    )
    pipe.to(DEVICE)

    if DEVICE == "cuda" or DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    elif DEVICE == "mps":
        # MPS doesn't support cpu_offload well
        pass
    else:
        print("경고: 알 수 없는 디바이스, 최적화 적용 안됨.")
        exit(1)

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return pipe

def generate_image(prompt, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image from text prompt and return for UI display."""
    global pipe

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        print(f"출력 크기: {width}x{height}")
        print(f"추론 스텝: {num_inference_steps}, 시드: {int(seed)}")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

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
    global demo

    # Print hardware specifications
    print_hardware_info()

    # Load model once at startup
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.2 Dev Klein Base 9B 이미지 생성기") as demo:
        gr.Markdown("# Flux.2 Dev Klein Base 9B Text-to-Image 생성기")
        gr.Markdown(f"텍스트 설명을 입력하여 이미지를 생성하세요. (Device: **{DEVICE.upper()}**)")

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
                            minimum=1.0,
                            maximum=10.0,
                            step=0.5,
                            value=4.0
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=4,
                            maximum=30,
                            step=1,
                            value=10
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
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
