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

#DEFAULT_PROMPT = "A sparkling-eyed Instagram-style young and cute skinny korean woman around from 35 to 39 year old, full-body photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner and she is posing like a model. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural and perfect. She is wearing a red bikini,  beautiful detailed body with perfect anatomy and perfect face-body ratio, beautiful gorgeous model, photorealistic, 4k, high quality, high resolution, beautiful body shape with perfect fingers and toes, perfect arms and legs structure, model pose, attractive face and body on the tropical sunny beach, cinematic and natural lighting, fashion vibe."

# DEFAULT_PROMPT = "photography, orange hue, korean cute girl model, wearing a very thin red bikini, looking at the right side, half-body, perfect anatomy, ,solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw"

#DEFAULT_PROMPT = "Photography, orange hue, korean cute girl model full-body shot, wearing a very thin red bikini, looking at the right side. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner and she is posing like a model. Her eyes are hazel, with a natural sparkle of happiness as she smiles, perfect anatomy and perfect face-body ratio, ,solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw"

DEFAULT_PROMPT= "Highly realistic, 4k, high-quality, high resolution, beautiful full body korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw."

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


# Define device type and data type
def get_device():
    """Detect available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


DEVICE = get_device()
if DEVICE == "cuda" or DEVICE == "mps":
    print(f"{DEVICE.upper()} detected, using bfloat16 for efficiency.")
    DTYPE = torch.bfloat16
elif DEVICE == "cpu":
    print("CPU detected, using float32 for better compatibility.")
    DTYPE = torch.float32
else:
    print("Unknown device, defaulting to float32.")
    DTYPE = torch.float32

pipe = None
current_model = None  # Track currently loaded model

MODEL_PATHS = {
    "4B": "black-forest-labs/FLUX.2-klein-4B",
    "9B": "black-forest-labs/FLUX.2-klein-9B",
}


def unload_model():
    """Unload the current model and free memory."""
    global pipe, current_model
    if pipe is not None:
        print("Unloading current model...")
        del pipe
        pipe = None
        current_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Model unloaded and memory freed!")


def load_model(model_size="9B"):
    """Load and initialize the Flux2Klein model with optimizations."""
    global pipe, current_model

    # If same model is already loaded, skip
    if current_model == model_size and pipe is not None:
        return f"모델 {model_size}이(가) 이미 로드되어 있습니다."

    # Unload existing model first
    unload_model()

    model_path = MODEL_PATHS.get(model_size, MODEL_PATHS["9B"])
    print(f"Loading {model_size} model on {DEVICE.upper()} (dtype: {DTYPE})...")

    try:
        pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=DTYPE)

        # Memory optimization settings based on device
        if DEVICE == "cuda" or DEVICE == "cpu":
            pipe.to(DEVICE)
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            pipe.enable_sequential_cpu_offload()
            print(
                "Enabled: model_cpu_offload, attention_slicing, sequential_cpu_offload"
            )
        else:
            # MPS-specific: no memory optimization
            pipe.to(DEVICE)
            print("MPS device detected, no memory optimization applied.")

        current_model = model_size
        print(f"{model_size} model loaded on {DEVICE.upper()}!")
        return f"✓ {model_size} 모델 로드 완료! ({DEVICE.upper()})"
    except Exception as e:
        return f"오류: 모델 로드 실패 - {str(e)}"


def generate_image(prompt, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image from text prompt and return for UI display."""
    global pipe

    if pipe is None:
        return None, "오류: 먼저 모델을 로드해주세요."

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
        output_path = f"{base_name}_{current_model}_{timestamp}_h{height}_w{width}_g{guidance_scale}_steps{num_inference_steps}_seed{seed}.png"
        image.save(output_path)
        print(f"이미지 저장됨: {output_path}")

        return image, f"✓ 이미지 생성 완료! 저장됨: {output_path}"

    except Exception as e:
        return None, f"오류: {str(e)}"


def main():
    # Create Gradio interface (model loaded on demand)
    with gr.Blocks(title="Flux.2 Dev Klein 4B/9B 이미지 생성기") as demo:
        gr.Markdown("# Flux.2 Dev Klein 4B/9B Text-to-Image 생성기")
        gr.Markdown(
            "모델을 선택하고 로드한 후, 텍스트 설명을 입력하여 이미지를 생성하세요."
        )

        with gr.Row():
            with gr.Column():
                # Model selection section
                with gr.Group():
                    gr.Markdown("### 모델 선택")
                    with gr.Row():
                        model_select = gr.Radio(
                            choices=["4B", "9B"],
                            value="9B",
                            label="모델 크기",
                            info="4B: 빠르고 가벼움 | 9B: 높은 품질",
                        )
                        model_status = gr.Textbox(
                            label="모델 상태",
                            value="모델이 로드되지 않았습니다.",
                            interactive=False,
                        )
                    load_btn = gr.Button("모델 로드", variant="primary")

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
                            value=0.5,
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=4,
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

        # Connect load button to model loading function
        load_btn.click(
            fn=load_model,
            inputs=[model_select],
            outputs=[model_status],
        )

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
