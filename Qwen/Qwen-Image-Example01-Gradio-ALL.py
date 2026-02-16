import torch
import platform
from diffusers import HiDreamImagePipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

DEFAULT_PROMPT = "A hyper-realistic, ultra-detailed full-body front-view depiction of beautiful Korean girl with no expression. Her dark brown hair is styled in a messy bun, with a few loose strands framing her face. Her flawless skin is illuminated by soft, even lighting, enhancing her natural contours without harsh shadows. She has a height of 168 cm, an ample chest with a naturally full bust (34F), a defined waist, and balanced hips. Her long, well-structured legs maintain their natural shape. She is wearing a red soft bra and thong. Pose: She stands with her feet flat on the ground, body fully visible from head to toe, arms resting naturally by her sides. The pose is neutral, ensuring an accurate full-body representation with true-to-life proportions. The background is plain and neutral, ensuring full focus on her body proportions. The soft, natural cinematic lighting enhances depth and realism while avoiding extreme contrast. Shot in HDR 4K, ultra-detailed textures, extreme photorealism, cinematic quality, with true-to-life proportions and no distortion, perfect anatomy, no extra fingers, no extra limbs, no extra hands, no extra feet, no extra legs, no extra toes."

POSITIVE_MAGIC = "Ultra HD, 4K, cinematic composition."

ASPECT_RATIOS = {
    "1:1 (1328x1328)": (1328, 1328),
    "16:9 (1664x928)": (1664, 928),
    "9:16 (928x1664)": (928, 1664),
    "4:3 (1472x1140)": (1472, 1140),
    "3:4 (1140x1472)": (1140, 1472),
    "3:2 (1584x1056)": (1584, 1056),
    "2:3 (1056x1584)": (1056, 1584),
}


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
    """Load and initialize the Qwen-Image model with optimizations."""
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
    pipe = HiDreamImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
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
    negative_prompt,
    width,
    height,
    true_cfg_scale,
    num_inference_steps,
    seed,
    append_magic,
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

        progress(0.0, desc="이미지 생성 준비 중...")
        print("이미지 생성 중...")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # Append positive magic prompt if enabled
        full_prompt = prompt
        if append_magic:
            full_prompt = prompt + " " + POSITIVE_MAGIC

        progress(0.05, desc="추론 시작...")
        print("추론 시작...")

        # Callback to report each inference step to Gradio progress bar and CLI status bar
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            # Map step progress to 0.05 ~ 0.90 range
            progress_val = 0.05 + ratio * 0.85
            progress(
                progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)"
            )

            # CLI status bar
            bar_len = 30
            filled = int(bar_len * ratio)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            speed = elapsed / current
            eta = speed * (steps - current)
            line = (
                f"  [{bar}] {current}/{steps} ({ratio*100:.0f}%) | "
                f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s | {speed:.2f}s/step"
            )
            print(f"\r{line:<80}", end="", flush=True)
            if current == steps:
                print()
            return callback_kwargs

        # Run the pipeline
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else " ",
            width=int(width),
            height=int(height),
            num_inference_steps=steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        ).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}_{DEVICE.upper()}_{int(width)}x{int(height)}_cfg{true_cfg_scale}_step{steps}_seed{int(seed)}.png"

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
    with gr.Blocks(title="Qwen-Image Text-to-Image Generator") as interface:
        gr.Markdown("# Qwen-Image Text-to-Image Generator")
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
                    placeholder="이미지에 대한 설명을 입력하세요",
                    info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다.",
                )
                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트",
                    value=" ",
                    lines=2,
                    placeholder="제거하고 싶은 요소를 입력하세요",
                    info="생성에서 제외하고 싶은 요소를 입력합니다. 특별히 제거할 요소가 없으면 비워두세요.",
                )
                append_magic = gr.Checkbox(
                    label="매직 프롬프트 추가 (Ultra HD, 4K, cinematic composition.)",
                    value=True,
                    info="프롬프트 끝에 품질 향상 문구를 자동으로 추가합니다.",
                )

                aspect_ratio = gr.Dropdown(
                    label="종횡비 (Aspect Ratio)",
                    choices=list(ASPECT_RATIOS.keys()),
                    value="16:9 (1664x928)",
                    info="이미지 종횡비를 선택하면 너비/높이가 자동으로 설정됩니다.",
                )

                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=4,
                        value=1664,
                        info="생성할 이미지의 너비를 지정합니다 (픽셀).",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=4,
                        value=928,
                        info="생성할 이미지의 높이를 지정합니다 (픽셀).",
                    )

                with gr.Row():
                    true_cfg_scale = gr.Slider(
                        label="True CFG Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=4.0,
                        info="모델이 프롬프트를 얼마나 따를지 제어합니다. 낮을수록 창의적, 높을수록 정확합니다. 권장: 3-7",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50,
                        info="이미지 생성 과정의 단계 수입니다. 높을수록 품질이 좋지만 시간이 더 걸립니다. 권장: 30-50",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 생성의 시작점입니다. 같은 시드를 사용하면 같은 결과를 얻습니다.",
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

        # Update width/height when aspect ratio is changed
        def update_aspect_ratio(ratio_name):
            w, h = ASPECT_RATIOS[ratio_name]
            return w, h

        aspect_ratio.change(
            fn=update_aspect_ratio,
            inputs=[aspect_ratio],
            outputs=[width, height],
        )

        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                width,
                height,
                true_cfg_scale,
                num_inference_steps,
                seed,
                append_magic,
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
