import re
import torch
import platform
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

DEFAULT_IMAGE = "Test03.jpg"

# Default values for each prompt section
DEFAULT_QUALITY = ""
DEFAULT_ANATOMY = ""
DEFAULT_SUBJECT = ""
DEFAULT_APPEARANCE = ""
DEFAULT_POSE = "She is lying down with both hands resting naturally on a pillow beside her head. One leg is slightly bent at the knee in a relaxed yet alluring pose, with a confident and sultry expression."
DEFAULT_OUTFIT = "She wears really small string simple white bra with minimum coverage and really small string simple white string panties with minimum coverage."
DEFAULT_SETTING = ""
DEFAULT_LIGHTING = ""
DEFAULT_CAMERA = ""


def load_default_image():
    """Load the default input image if it exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, DEFAULT_IMAGE)
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    return None


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    quality, anatomy, subject, appearance, pose, outfit, setting, lighting, camera
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [
        quality,
        anatomy,
        subject,
        appearance,
        pose,
        outfit,
        setting,
        lighting,
        camera,
    ]
    combined = ", ".join(normalize_spacing(s) for s in sections if s and s.strip())
    return combined


def round_to_64(value: int) -> int:
    """Round value to the nearest multiple of 64, minimum 256."""
    return max(256, round(value / 64) * 64)


def get_image_dimensions(image):
    """Read uploaded image size and return width/height rounded to 64."""
    if image is None:
        return 768, 1536
    w, h = image.size
    return round_to_64(w), round_to_64(h)


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


signal.signal(signal.SIGINT, signal_handler)


def load_model(device_name=None):
    """Load and initialize the Flux2Klein model with optimizations."""
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
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B",
        torch_dtype=DTYPE,
    )

    if DEVICE == "cuda" or DEVICE == "cpu":
        pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(
            "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing"
        )
    elif DEVICE == "mps":
        pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def on_image_upload(image):
    """입력 이미지 업로드 시 출력 크기 자동 설정."""
    if image is None:
        return 768, 1536, "이미지를 업로드하면 원본 크기가 표시됩니다."
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    w, h = image.size
    rw, rh = round_to_64(w), round_to_64(h)
    info = f"원본 크기: {w} × {h} px  →  출력 크기: {rw} × {rh} px (64 배수로 반올림)"
    return rw, rh, info


def generate_image(
    input_image,
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

    if input_image is None:
        input_image = load_default_image()
    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)

        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="이미지 편집 준비 중...")
        print("이미지 편집 준비 중...")

        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        progress(0.05, desc="추론 시작...")
        print("추론 시작...")

        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.90
            progress(
                progress_val,
                desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)",
            )

            bar_len = 30
            filled = int(bar_len * ratio)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            speed = elapsed / current
            eta = speed * (steps - current)
            line = (
                f"  [{bar}] {current}/{steps} ({ratio*100:.0f}%) | "
                f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s | {speed:.2f}s/step"
            )
            print(f"\r{line}\033[K", end="", flush=True)
            if current == steps:
                print()
            return callback_kwargs

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                image=input_image,
                width=int(width),
                height=int(height),
                guidance_scale=guidance_scale,
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

        print(f"이미지 편집 완료! 소요 시간: {elapsed:.1f}초")
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

    # Print hardware specifications
    print_hardware_info()

    # Auto-load model on startup with detected device
    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    # Load default image and compute initial dimensions
    _default_img = load_default_image()
    _init_w, _init_h = get_image_dimensions(_default_img)
    _default_info = (
        f"기본 이미지: {DEFAULT_IMAGE} ({_default_img.size[0]} × {_default_img.size[1]} px)"
        if _default_img is not None
        else "이미지를 업로드하면 원본 크기가 표시됩니다."
    )

    # Create Gradio interface
    with gr.Blocks(
        title="Flux.2 Klein 9B Image-to-Image Generator",
    ) as interface:
        gr.Markdown("# Flux.2 Klein 9B Image-to-Image Generator")
        gr.Markdown(
            f"입력 이미지를 텍스트 프롬프트를 사용하여 편집합니다."
            f" (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            # Left column: Model loading + Input image + Prompt sections
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

                gr.Markdown("### 입력 이미지")
                input_image = gr.Image(
                    label="입력 이미지 (업로드하면 출력 크기가 자동 설정됩니다)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=800,
                    value=_default_img,
                )
                image_info = gr.Textbox(
                    label="이미지 정보",
                    value=_default_info,
                    interactive=False,
                )

                gr.Markdown("### 프롬프트 구성")
                prompt_quality = gr.Textbox(
                    label="1. 품질/해상도 (Quality & Resolution)",
                    value=DEFAULT_QUALITY,
                    lines=2,
                    placeholder="예: 4k, ultra-detailed, photorealistic",
                    info="이미지의 품질, 해상도, 스타일 관련 키워드입니다. 프롬프트 맨 앞에 위치합니다.",
                )
                prompt_anatomy = gr.Textbox(
                    label="2. 신체 구조 (Anatomy)",
                    value=DEFAULT_ANATOMY,
                    lines=2,
                    placeholder="예: bad anatomy, extra limbs, deformed body",
                    info="신체 구조 관련 제외 요소들입니다. 최종 프롬프트에 포함됩니다.",
                )
                prompt_subject = gr.Textbox(
                    label="3. 주제 (Subject)",
                    value=DEFAULT_SUBJECT,
                    lines=2,
                    placeholder="예: A stunning photorealistic image of a young woman on a beach",
                    info="이미지의 주제와 전체적인 장면을 한 문장으로 설명합니다.",
                )
                prompt_appearance = gr.Textbox(
                    label="4. 외모 (Appearance)",
                    value=DEFAULT_APPEARANCE,
                    lines=2,
                    placeholder="예: A beautiful woman with long strawberry blonde hair",
                    info="인물의 외모, 얼굴, 머리카락, 나이 등을 설명합니다.",
                )
                prompt_pose = gr.Textbox(
                    label="5. 포즈/구도 (Pose & Composition)",
                    value=DEFAULT_POSE,
                    lines=2,
                    placeholder="예: seated on rooftop ledge, gazing into distance",
                    info="자세, 시선 방향, 카메라 앵글, 촬영 구도를 설명합니다.",
                )
                prompt_outfit = gr.Textbox(
                    label="6. 의상 (Outfit)",
                    value=DEFAULT_OUTFIT,
                    lines=2,
                    placeholder="예: deep burgundy satin slip dress",
                    info="의상, 액세서리, 착용한 아이템을 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="7. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    placeholder="예: rooftop terrace at twilight, city skyline",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="8. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    placeholder="예: golden hour, city glow, cinematic rim light",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="9. 카메라 설정 (Camera Settings)",
                    value=DEFAULT_CAMERA,
                    lines=2,
                    placeholder="예: Sony A7R V, 85mm f/1.8, ISO 400",
                    info="카메라 기종, 렌즈, ISO, 셔터 스피드, 조리개, 피사계 심도 등을 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            DEFAULT_QUALITY,
                            DEFAULT_ANATOMY,
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
                    prompt_quality,
                    prompt_anatomy,
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

            # Right column: Parameters + Generate + Output
            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="출력 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=_init_w,
                        info="출력 이미지 너비 (픽셀). 입력 이미지 업로드 시 자동 설정되며 수동 변경도 가능합니다.",
                    )
                    height = gr.Slider(
                        label="출력 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=_init_h,
                        info="출력 이미지 높이 (픽셀). 입력 이미지 업로드 시 자동 설정되며 수동 변경도 가능합니다.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        info="Klein 권장: 1.0. 낮으면 창의적, 높으면 정확. 권장 범위: 0.5-1.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=4,
                        info="Klein 권장: 4. 높으면 품질 향상, 시간 증가. 권장: 4-12",
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
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        # 입력 이미지 업로드 시 출력 크기 자동 설정 (수동 변경도 가능)
        input_image.change(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[width, height, image_info],
        )

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
        allowed_paths=[os.path.dirname(os.path.abspath(__file__))],
        js="document.addEventListener('keydown',function(e){if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();}})",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
