import torch
import platform
from diffusers import FluxPipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

DEFAULT_PROMPT = "The image is a high-quality, photorealistic cosplay portrait of a young Korean woman with a soft, idol aesthetic. Physical Appearance: Face: She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera: She has long, straight jet-black hair with thick, straight-cut bangs (fringe) that frame her face. Attire (Blue & White Bunny Theme): Headwear: She wears tall, upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base, accented with a small white bow. Outfit: She wears a unique blue denim-textured bodysuit. It features a front zipper, silver buttons, and thin silver chains draped across the chest. The sides are constructed from semi-sheer white lace. Accessories: Around her neck is a blue bow tie attached to a white collar. She wears long, white floral lace fingerless sleeves that extend past her elbows, finished with blue cuffs and small black decorative ribbons. Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows. Pose: She is standing gracefully in front of the edge of a light-colored, vintage-style bed or cushioned bench. Her body is slightly angled toward the camera, creating a soft and inviting posture. Setting & Background: Location: A bright, high-key studio set designed to look like a clean, airy bedroom. Background: The background is dominated by large windows with white vertical blinds or curtains, allowing soft, diffused natural-looking light to flood the scene. The background is softly blurred (bokeh). Lighting: The lighting is bright, soft, and even, minimizing harsh shadows and giving the skin a glowing, porcelain appearance. Flux Prompt Prompt: A photorealistic, high-quality cosplay portrait of a beautiful Korean woman dressed in a blue and white bunny girl outfit. She has long straight black hair with hime-cut bangs and vibrant blue eyes. She wears tall blue bunny ears with white lace trim, a blue denim-textured bodysuit with a front zipper and white lace side panels, a blue bow tie, and long white lace sleeves. She is standing in front of a white bed in a bright, sun-drenched room with soft-focus white curtains. She is looking at the camera with a soft, innocent expression.8k resolution, high-key lighting, cinematic soft focus, detailed textures of denim and lace, gravure photography style. Key Stylistic Keywords Blue bunny girl, denim cosplay, white lace, high-key lighting, blue contact lenses, black hair with bangs, fishnet stockings, airy atmosphere, photorealistic, innocent and alluring, studio photography."


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


# Global variables
DEVICE, DTYPE = get_device_and_dtype()
pipe = None
interface = None


def get_available_devices():
    """Return list of available device choices.
    - CUDA + CPU: both selectable
    - MPS only (no CUDA): MPS only
    - No GPU: CPU only
    """
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
        devices.append("cpu")
    elif torch.backends.mps.is_available():
        devices.append("mps")
    else:
        devices.append("cpu")
    return devices


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


def cleanup():
    """Release all resources and clear memory."""
    global pipe, interface
    print("\n리소스 정리 중...")

    if interface is not None:
        try:
            interface.close()
            print("Gradio 서버 종료됨")
        except Exception:
            pass
        interface = None

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


def load_model(device_name=None):
    """Load and initialize the Flux model with optimizations."""
    global pipe, DEVICE, DTYPE

    if device_name is not None:
        DEVICE = device_name
        DTYPE = torch.bfloat16 if device_name in ("cuda", "mps") else torch.float32

    # Release previous model if loaded
    if pipe is not None:
        print("기존 모델 해제 중...")
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"모델 로딩 중... (Device: {DEVICE}, dtype: {DTYPE})")
    print("T5-XXL 텍스트 인코더만 사용합니다. (CLIP 비활성화)")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    # Enable memory optimizations based on device
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(
            "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CUDA)"
        )
    elif DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(
            "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CPU)"
        )
    elif DEVICE == "mps":
        # MPS doesn't support cpu_offload well
        print("No memory optimizations applied for MPS device.")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    strength,
    max_sequence_length,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe

    if pipe is None:
        return (
            None,
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중...")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # Encode prompt using T5 only (CLIP is disabled)
        text_inputs = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=int(max_sequence_length),
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = pipe.text_encoder_2(
                text_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=DTYPE)

        # Zero pooled embeddings (normally from CLIP, not needed with T5-only)
        pooled_prompt_embeds = torch.zeros(
            1, 768, dtype=DTYPE, device=prompt_embeds.device
        )

        progress(0.05, desc="추론 시작...")
        print("추론 시작...")

        # Callback to report each inference step to Gradio progress bar and CLI status bar
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            # Map step progress to 0.05 ~ 0.90 range
            progress_val = 0.05 + ratio * 0.85
            progress(progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)")

            # CLI status bar
            bar_len = 30
            filled = int(bar_len * ratio)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            speed = elapsed / current
            eta = speed * (steps - current)
            print(
                f"\r  [{bar}] {current}/{steps} ({ratio*100:.0f}%) | "
                f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s | {speed:.2f}s/step",
                end="",
                flush=True,
            )
            if current == steps:
                print()
            return callback_kwargs

        # Build pipeline kwargs with pre-computed embeddings
        pipe_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        # Run the pipeline
        image = pipe(**pipe_kwargs).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}_{DEVICE.upper()}_{width}x{height}_gs{guidance_scale}_step{steps}_seed{int(seed)}_str{strength}_msl{int(max_sequence_length)}.png"

        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")
        print(f"이미지가 저장되었습니다 : {filename}")
        image.save(filename)

        progress(1.0, desc="완료!")
        return image, f"✓ 완료! ({elapsed:.1f}초) 저장됨: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    # Print hardware specifications
    print_hardware_info()

    # Auto-load model on startup with detected device
    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.1-dev Text-to-Image Generator") as interface:
        gr.Markdown("# Flux.1-dev Text-to-Image Generator")
        gr.Markdown(
            f"AI를 사용하여 텍스트에서 이미지를 생성합니다. (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                device_selector = gr.Radio(
                    label="디바이스 선택",
                    choices=get_available_devices(),
                    value=DEVICE,
                    info="모델을 실행할 디바이스를 선택하세요.",
                )
                load_model_btn = gr.Button("모델 로드", variant="secondary")
                device_status = gr.Textbox(
                    label="모델 상태",
                    value=(
                        f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"
                        if pipe is not None
                        else "모델이 로드되지 않았습니다. 디바이스를 선택하고 '모델 로드' 버튼을 눌러주세요."
                    ),
                    interactive=False,
                )

                # Input parameters
                prompt = gr.Textbox(
                    label="프롬프트",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    placeholder="이미지에 대한 설명을 입력하세요 (77단어 이하 권장)",
                    info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다. 예: '여자, 미소, 해변, 빨간 비키니'",
                )
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=768,
                        info="생성할 이미지의 너비를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1536,
                        info="생성할 이미지의 높이를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=4.0,
                        info="모델이 프롬프트를 얼마나 따를지 제어합니다. 낮을수록 창의적, 높을수록 정확합니다. 권장: 4-15",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28,
                        info="이미지 생성 과정의 단계 수입니다. 높을수록 품질이 좋지만 시간이 더 걸립니다. 권장: 20-28",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 생성의 시작점입니다. 같은 시드를 사용하면 같은 결과를 얻습니다.",
                    )
                    strength = gr.Slider(
                        label="강도",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                        info="생성 모델의 강도를 제어합니다. 낮을수록 다양한 결과, 높을수록 일관성 있는 결과입니다.",
                    )

                with gr.Row():
                    max_sequence_length = gr.Slider(
                        label="최대 시퀀스 길이",
                        minimum=64,
                        maximum=512,
                        step=64,
                        value=512,
                        info="텍스트 인코더의 최대 시퀀스 길이입니다. 긴 프롬프트를 사용할 경우 높은 값이 필요합니다.",
                    )

                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        # Load model when button is clicked
        load_model_btn.click(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        )

        # Auto-load model when device is changed
        device_selector.change(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        )

        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                strength,
                max_sequence_length,
            ],
            outputs=[output_image, output_message],
        )

    # Launch the interface
    interface.launch(inbrowser=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
