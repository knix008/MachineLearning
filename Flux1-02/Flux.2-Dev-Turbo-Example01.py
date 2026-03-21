import re
import torch
import platform
from diffusers import Flux2Pipeline
from datetime import datetime
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

# Default values for each prompt section
DEFAULT_SUBJECT = "The image is a high-quality, photorealistic cosplay portrait of a young Asian woman with a soft, idol aesthetic."

DEFAULT_APPEARANCE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera with her index finger lightly touching her chin. She has long, straight jet-black hair with thick, straight-cut bangs (fringe) that frame her face."

DEFAULT_POSE = "Sitting gracefully on the edge of a light-colored, vintage-style bed or cushioned bench. Her body is slightly angled toward the camera, creating a soft and inviting posture."

DEFAULT_OUTFIT = "Tall upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband, accented with a small white bow. Blue denim-textured bodysuit with a front zipper, silver buttons, and thin silver chains draped across the chest, with semi-sheer white lace sides. Blue bow tie attached to a white collar. Long white floral lace fingerless sleeves with blue cuffs and small black decorative ribbons. White fishnet stockings held up by blue and white ruffled lace garters with small white bows."

DEFAULT_SETTING = "Bright, high-key studio set designed to look like a clean, airy bedroom. Large windows with white vertical blinds or curtains, allowing soft diffused natural-looking light to flood the scene. Softly blurred background (bokeh)."

DEFAULT_LIGHTING = "Bright, soft, and even lighting minimizing harsh shadows. Skin has a glowing, porcelain appearance. High-key lighting, cinematic soft focus."

DEFAULT_CAMERA = "8K resolution, tack-sharp focus, detailed textures of denim and lace, gravure photography style."


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    subject, appearance, pose, outfit, setting, lighting, camera
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [subject, appearance, pose, outfit, setting, lighting, camera]
    combined = ", ".join(normalize_spacing(s) for s in sections if s and s.strip())
    return combined


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
    """Return list of available device choices."""
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

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OS 버전: {platform.version()}")
    print(f"아키텍처: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    print("-" * 60)
    print("CPU 정보")
    print("-" * 60)
    print(f"프로세서: {platform.processor()}")
    print(f"물리 코어: {psutil.cpu_count(logical=False)}")
    print(f"논리 코어: {psutil.cpu_count(logical=True)}")

    mem = psutil.virtual_memory()
    print("-" * 60)
    print("메모리 정보")
    print("-" * 60)
    print(f"총 RAM: {mem.total / (1024**3):.1f} GB")
    print(f"사용 가능: {mem.available / (1024**3):.1f} GB")
    print(f"사용률: {mem.percent}%")

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
    """Load and initialize the FLUX.2-dev Turbo pipeline with LoRA weights."""
    global pipe, DEVICE, DTYPE

    if device_name is not None:
        DEVICE = device_name
        DTYPE = torch.bfloat16 if device_name in ("cuda", "mps") else torch.float32

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
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        torch_dtype=DTYPE,
    )

    pipe.load_lora_weights(
        "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
    )

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
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    image_format,
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

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Use turbo sigmas when steps fit within the precomputed table
        sigmas = TURBO_SIGMAS[:steps] if steps <= len(TURBO_SIGMAS) else None

        progress(0.05, desc="추론 시작...")

        # Callback to report each inference step to Gradio progress bar and CLI status bar
        last_step_time = [start_time]

        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            now = time.time()
            elapsed = now - start_time
            step_time = now - last_step_time[0]
            last_step_time[0] = now
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.85
            progress(
                progress_val,
                desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)",
            )

            bar_len = 40
            filled = int(bar_len * ratio)
            bar = "█" * filled + "░" * (bar_len - filled)
            avg_speed = elapsed / current
            eta = avg_speed * (steps - current)
            sys.stdout.write(
                f"\r추론 진행: |{bar}| {current}/{steps} "
                f"[{elapsed:.1f}s elapsed, ETA {eta:.1f}s, "
                f"{step_time:.2f}s/step, avg {avg_speed:.2f}s/step]"
            )
            sys.stdout.flush()
            if current == steps:
                print()

            return callback_kwargs

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                height=int(height),
                width=int(width),
                num_inference_steps=steps,
                generator=generator,
                callback_on_step_end=step_callback,
            ).images[0]

        progress(0.95, desc="이미지 저장 중...")

        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        filename = (
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{int(width)}x{int(height)}"
            f"_gs{guidance_scale}_step{steps}_seed{int(seed)}.{ext}"
        )

        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")
        print(f"이미지가 저장되었습니다 : {filename}")
        if image_format == "JPEG":
            image.save(filename, format="JPEG", quality=100, subsampling=0)
        else:
            image.save(filename)

        progress(1.0, desc="완료!")
        return (
            image,
            f"✓ 완료! ({elapsed:.1f}초) | 저장됨: {filename}",
        )
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()

    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    with gr.Blocks(
        title="FLUX.2-dev Turbo Text-to-Image Generator",
    ) as interface:
        gr.Markdown("# FLUX.2-dev Turbo Text-to-Image Generator")
        gr.Markdown(
            f"FLUX.2-dev + Turbo LoRA 모델을 사용하여 텍스트에서 이미지를 생성합니다."
            f" (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            # Left column: Model loading + Prompt sections
            with gr.Column(scale=1):
                gr.Markdown("### 모델 설정")
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
                        else "모델이 로드되지 않았습니다."
                        " 디바이스를 선택하고 '모델 로드' 버튼을 눌러주세요."
                    ),
                    interactive=False,
                )

                gr.Markdown("### 프롬프트 구성")
                prompt_subject = gr.Textbox(
                    label="1. 주제/대상 (Subject)",
                    value=DEFAULT_SUBJECT,
                    lines=2,
                    placeholder="예: A high-quality photorealistic portrait of a young Asian woman",
                    info="이미지의 주된 주제나 대상을 설명합니다.",
                )
                prompt_appearance = gr.Textbox(
                    label="2. 외모 (Appearance)",
                    value=DEFAULT_APPEARANCE,
                    lines=2,
                    placeholder="예: Fair skin, blue contact lenses, long black hair with bangs",
                    info="인물의 외모, 얼굴, 머리카락, 나이 등을 설명합니다.",
                )
                prompt_pose = gr.Textbox(
                    label="3. 포즈/구도 (Pose & Composition)",
                    value=DEFAULT_POSE,
                    lines=2,
                    placeholder="예: Sitting gracefully, body slightly angled toward camera",
                    info="자세, 시선 방향, 카메라 앵글, 촬영 구도를 설명합니다.",
                )
                prompt_outfit = gr.Textbox(
                    label="4. 의상 (Outfit)",
                    value=DEFAULT_OUTFIT,
                    lines=2,
                    placeholder="예: Blue bunny ears, denim bodysuit, white lace sleeves",
                    info="의상, 액세서리, 착용한 아이템을 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="5. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    placeholder="예: Bright airy studio bedroom, white curtains, soft bokeh",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="6. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    placeholder="예: High-key soft lighting, porcelain skin glow",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="7. 카메라 설정 (Camera Settings)",
                    value=DEFAULT_CAMERA,
                    lines=2,
                    placeholder="예: 8K, tack-sharp focus, gravure photography style",
                    info="카메라 기종, 렌즈, 해상도, 촬영 스타일 등을 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            DEFAULT_SUBJECT,
                            DEFAULT_APPEARANCE,
                            DEFAULT_POSE,
                            DEFAULT_OUTFIT,
                            DEFAULT_SETTING,
                            DEFAULT_LIGHTING,
                            DEFAULT_CAMERA,
                        ),
                        lines=4,
                        interactive=False,
                        info="위 섹션들이 자동으로 합쳐진 최종 프롬프트입니다.",
                    )
                prompt_sections = [
                    prompt_subject,
                    prompt_appearance,
                    prompt_pose,
                    prompt_outfit,
                    prompt_setting,
                    prompt_lighting,
                    prompt_camera,
                ]
                for section in prompt_sections:
                    section.change(
                        fn=combine_prompt_sections,
                        inputs=prompt_sections,
                        outputs=[combined_prompt],
                    )

            # Right column: Parameters (top) + Image generation (bottom)
            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=768,
                        info="이미지 너비 (픽셀). 64의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=512,
                        maximum=2048,
                        step=64,
                        value=1536,
                        info="이미지 높이 (픽셀). 64의 배수.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=2.5,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. 권장: 2.5",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=4,
                        maximum=20,
                        step=1,
                        value=8,
                        info="생성 단계 수. Turbo 권장: 4-8",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    image_format = gr.Radio(
                        label="이미지 포맷",
                        choices=["JPEG", "PNG"],
                        value="JPEG",
                        info="JPEG: quality 100 (4:4:4), PNG: 무손실 압축.",
                    )

                gr.Markdown("---")
                gr.Markdown("### 이미지 생성")
                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")
                output_image = gr.Image(label="생성된 이미지", height=700)
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
                combined_prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                image_format,
            ],
            outputs=[output_image, output_message],
        )

    # Launch the interface
    interface.launch(
        inbrowser=True,
        js="document.addEventListener('keydown',function(e){if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();}})",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
