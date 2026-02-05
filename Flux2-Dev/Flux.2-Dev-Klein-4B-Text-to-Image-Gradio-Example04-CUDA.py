import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import platform
import subprocess
import gc
import atexit
import signal
import sys
import gradio as gr


def print_hardware_info():
    """Print hardware information at startup."""
    print("=" * 60)
    print("HARDWARE INFORMATION")
    print("=" * 60)

    # System info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")

    # CPU info (Windows-specific)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "wmic",
                    "cpu",
                    "get",
                    "name,numberofcores,numberoflogicalprocessors",
                    "/format:list",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip() and "=" in line:
                    print(f"CPU {line.strip()}")
        except Exception:
            pass

        # RAM info
        try:
            result = subprocess.run(
                ["wmic", "memorychip", "get", "capacity", "/format:list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            total_ram = 0
            for line in result.stdout.strip().split("\n"):
                if line.strip().startswith("Capacity="):
                    total_ram += int(line.split("=")[1])
            if total_ram > 0:
                print(f"RAM Total: {total_ram / (1024**3):.1f} GB")
        except Exception:
            pass

    # GPU info via PyTorch
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"GPU {i} Memory: {props.total_memory / (1024**3):.1f} GB")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available")
    else:
        print("GPU: Not available (CPU only)")

    print("=" * 60)


print_hardware_info()


def cleanup():
    """Release all resources before exit."""
    global pipe
    print("Releasing resources...")
    try:
        del pipe
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Resources released!")


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle keyboard interrupt."""
    print("\nKeyboard interrupt received...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


DEFAULT_PROMPT = "A sparkling-eyed Instagram-style young  and cute  korean woman wearing a red bikini, beautiful detailed body with perfect anatomy and perfect arms and legs structure, perfect fingers and toes, beautiful gorgeous model, photorealistic, 4k, high quality, high resolution, beautiful body, attractive pose, attractive face and body --niji 5 --ar 9:16"

# Global variables for model
DEVICE = "cuda"
DTYPE = torch.bfloat16
pipe = None


def load_model():
    """Load and initialize the Flux2Klein model with optimizations."""
    global pipe

    print("모델 로딩 중...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", torch_dtype=DTYPE
    )
    pipe = pipe.to(DEVICE)

    # Memory optimization
    # pipe.enable_model_cpu_offload() # CUDA에서 CPU RAM을 일부 사용
    # pipe.enable_attention_slicing() # 안쓰면 GPU 메모리를 더 사용함(속)
    # pipe.enable_sequential_cpu_offload() # 안쓰면 CUDA에서 더 빠름(4 추론스텝에서 1초 단축)

    print("모델 로딩 완료!")
    return pipe


def generate_image(prompt, height, width, guidance_scale, num_inference_steps, seed):
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

        print(f"이미지 생성 중... (steps: {num_inference_steps})")
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
                    lines=5,
                )

                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        height_input = gr.Slider(
                            label="높이 (Height)",
                            info="생성할 이미지의 높이 (픽셀)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1536,
                        )
                        width_input = gr.Slider(
                            label="너비 (Width)",
                            info="생성할 이미지의 너비 (픽셀)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=612,
                        )

                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0,
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=4,
                            maximum=50,
                            step=1,
                            value=10,
                        )

                    seed_input = gr.Slider(
                        label="시드 (Seed)",
                        info="재현성을 위한 난수 시드 (0: 랜덤)",
                        minimum=-1,
                        maximum=1000,
                        step=1,
                        value=42,
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
                seed_input,
            ],
            outputs=[image_output, status_output],
        )

    # Launch the interface
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
