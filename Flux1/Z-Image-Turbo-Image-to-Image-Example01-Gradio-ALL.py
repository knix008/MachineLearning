import torch
import platform
from diffusers import ZImageImg2ImgPipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# https://prompthero.com/prompt/a885d64532b-flux-flux-11-pro-ultra-a-professional-photo-of-a-exceptionally-of-the-most-beautiful-russian-girl-in-the-world-age-18-futuristicmasterpiece-realistic-perfect

# Default values for each prompt section
DEFAULT_QUALITY = ""
DEFAULT_NEGATIVE = ""
DEFAULT_APPEARANCE = ""
DEFAULT_OUTFIT = "She is wearing a tiny black bikini. Remove the black belt."
DEFAULT_POSE = ""
DEFAULT_SETTING = ""
DEFAULT_LIGHTING = ""
DEFAULT_CAMERA = ""

# ── Default input image ───────────────────────────────────────────────────────
DEFAULT_INPUT_IMAGE_PATH = "Test01.jpg"  # 기본 입력 이미지 경로. 예: "default_input.png"
_default_input_img = None
_default_img_w, _default_img_h = 768, 1024  # 기본 이미지가 없을 때 폴백 크기
if DEFAULT_INPUT_IMAGE_PATH and os.path.isfile(DEFAULT_INPUT_IMAGE_PATH):
    try:
        _default_input_img = Image.open(DEFAULT_INPUT_IMAGE_PATH).convert("RGB")
        _default_img_w, _default_img_h = _default_input_img.size
        print(
            f"기본 입력 이미지 로드됨: {DEFAULT_INPUT_IMAGE_PATH}"
            f" ({_default_img_w}x{_default_img_h})"
        )
    except Exception as e:
        print(f"기본 입력 이미지 로드 실패: {e}")


def combine_prompt_sections(
    quality, negative, appearance, outfit, pose, setting, lighting, camera
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [quality, negative, appearance, outfit, pose, setting, lighting, camera]
    # Filter out empty sections and join with ', '
    combined = ", ".join(s.strip() for s in sections if s and s.strip())
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
    """Load and initialize the Z-Image model with optimizations."""
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
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
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
        print("MPS 최적화 활성화 안함")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    input_image,
    width,
    height,
    strength,
    prompt,
    negative_prompt,
    guidance_scale,
    num_inference_steps,
    seed,
    cfg_normalization,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe

    if pipe is None:
        return (
            None,
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="이미지 생성 준비 중...")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

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
            # Map step progress to 0.05 ~ 0.90 range
            progress_val = 0.05 + ratio * 0.85
            progress(
                progress_val,
                desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)",
            )

            # CLI status bar
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
                print()  # newline after completion

            return callback_kwargs

        # Resize input image to specified dimensions
        resized_image = input_image.resize((int(width), int(height)), Image.LANCZOS)

        # Run the pipeline
        image = pipe(
            image=resized_image,
            strength=strength,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            cfg_normalization=cfg_normalization,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        ).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        w, h = resized_image.size
        filename = (
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{w}x{h}"
            f"_gs{guidance_scale}_step{steps}_str{strength}_cfgnorm{cfg_normalization}_seed{int(seed)}.png"
        )

        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")
        print(f"이미지가 저장되었습니다 : {filename}")
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

    # Print hardware specifications
    print_hardware_info()

    # Auto-load model on startup with detected device
    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    # Create Gradio interface
    with gr.Blocks(
        title="Z-Image Image-to-Image Generator",
    ) as interface:
        gr.Markdown("# Z-Image Image-to-Image Generator")
        gr.Markdown(
            f"Tongyi-MAI/Z-Image 모델을 사용하여 입력 이미지를 변환합니다."
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
                prompt_quality = gr.Textbox(
                    label="1. 품질/해상도 (Quality & Resolution)",
                    value=DEFAULT_QUALITY,
                    lines=2,
                    placeholder="예: 4k, ultra-detailed, photorealistic",
                    info="이미지의 품질, 해상도, 스타일 관련 키워드입니다.",
                )
                prompt_negative = gr.Textbox(
                    label="네거티브 프롬프트 (부정적 요소)",
                    value=DEFAULT_NEGATIVE,
                    lines=2,
                    placeholder="예: extra hands, bad anatomy, low quality",
                    info="생성에서 제외할 요소를 기술합니다. 이것은 별도의 파라미터로 전달됩니다.",
                )
                prompt_appearance = gr.Textbox(
                    label="2. 외모 (Appearance)",
                    value=DEFAULT_APPEARANCE,
                    lines=2,
                    placeholder="예: A beautiful Korean girl with long black hair",
                    info="인물의 외모, 얼굴, 머리카락, 나이 등을 설명합니다.",
                )
                prompt_outfit = gr.Textbox(
                    label="3. 의상 (Outfit)",
                    value=DEFAULT_OUTFIT,
                    lines=2,
                    placeholder="예: in a red bikini, wearing a white dress",
                    info="의상, 액세서리, 착용한 아이템을 설명합니다.",
                )
                prompt_pose = gr.Textbox(
                    label="4. 포즈/구도 (Pose & Composition)",
                    value=DEFAULT_POSE,
                    lines=2,
                    placeholder="예: standing, looking over shoulder, full body",
                    info="자세, 시선 방향, 카메라 앵글, 촬영 구도를 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="5. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    placeholder="예: on a boardwalk at sunset, calm ocean",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="6. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    placeholder="예: golden hour, soft glow, cinematic lighting",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="7. 카메라 설정 (Camera Settings)",
                    value=DEFAULT_CAMERA,
                    lines=2,
                    placeholder="예: Canon EOS R5, 85mm f/1.4, ISO 100, shallow DOF",
                    info="카메라 기종, 렌즈, ISO, 셔터 스피드, 조리개, 피사계 심도 등을 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            DEFAULT_QUALITY,
                            "",  # Negative prompt is separate
                            DEFAULT_APPEARANCE,
                            DEFAULT_OUTFIT,
                            DEFAULT_POSE,
                            DEFAULT_SETTING,
                            DEFAULT_LIGHTING,
                            DEFAULT_CAMERA,
                        ),
                        lines=4,
                        interactive=False,
                        info="위 섹션들이 자동으로 합쳐진 최종 프롬프트입니다 (네거티브 프롬프트 제외).",
                    )
                prompt_sections = [
                    prompt_quality,
                    prompt_appearance,
                    prompt_outfit,
                    prompt_pose,
                    prompt_setting,
                    prompt_lighting,
                    prompt_camera,
                ]

                def update_combined(*sections):
                    return combine_prompt_sections(sections[0], "", *sections[1:])

                for section in prompt_sections:
                    section.change(
                        fn=update_combined,
                        inputs=prompt_sections,
                        outputs=[combined_prompt],
                    )

            # Right column: Parameters (top) + Image generation (bottom)
            with gr.Column(scale=1):
                gr.Markdown("### 입력 이미지")
                input_image = gr.Image(
                    label="입력 이미지 (변환할 원본 이미지를 업로드하세요)",
                    type="pil",
                    height=400,
                    value=_default_input_img,
                )

                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=64,
                        maximum=2048,
                        step=64,
                        value=_default_img_w,
                        info="출력 이미지 너비 (픽셀). 입력 이미지를 이 크기로 리사이즈 후 파이프라인 실행.",
                    )
                    height = gr.Slider(
                        label="Height",
                        minimum=64,
                        maximum=2048,
                        step=64,
                        value=_default_img_h,
                        info="출력 이미지 높이 (픽셀). 입력 이미지를 이 크기로 리사이즈 후 파이프라인 실행.",
                    )

                def update_size_from_image(img):
                    if img is None:
                        return gr.update(), gr.update()
                    w, h = img.size
                    return gr.update(value=w), gr.update(value=h)

                input_image.change(
                    fn=update_size_from_image,
                    inputs=[input_image],
                    outputs=[width, height],
                )

                strength = gr.Slider(
                    label="Strength (변환 강도)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.50,
                    info="원본 이미지에서 얼마나 벗어날지. 높을수록 많이 변환. 권장: 0.6~0.85",
                )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=0.0, # Guidance Should be 0 for Z-Image Turbo model.
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. 권장: 4.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=9, # It is used in Z-Image Turbo model example code.
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 25",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        minimum=0,
                        maximum=1000,
                        value=42,
                        precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    cfg_normalization = gr.Checkbox(
                        label="CFG Normalization",
                        value=False,
                        info="CFG 정규화 사용 여부.",
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
                input_image,
                width,
                height,
                strength,
                combined_prompt,
                prompt_negative,
                guidance_scale,
                num_inference_steps,
                seed,
                cfg_normalization,
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
