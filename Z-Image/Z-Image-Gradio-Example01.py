import torch
import platform
from diffusers import ZImagePipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# Default values for each prompt section
DEFAULT_QUALITY = "A professional photo, (futuristic), (masterpiece), (realistic), (perfect anatomy), 8k, highly detailed, full length frame, High detail RAW color art, sharp focus, hyper realism."

DEFAULT_NEGATIVE = "Extra hands, extra legs, extra feet, extra arms, Waist Pleats, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, wet, acnes, skin blemishes, age spot, manboobs, backlight, mutated hands, (poorly drawn hands:1.33), blurry, (bad anatomy:1.21), (bad proportions:1.33), extra limbs, (disfigured:1.33), (more than 2 nipples:1.33), (missing arms:1.33), (extra legs:1.33), (fused fingers:1.61), (too many fingers:1.61), (unclear eyes:1.33), lowers, bad hands, missing fingers, extra digit, (futa:1.1), bad hands, missing fingers, (cleft chin:1.3)"

DEFAULT_APPEARANCE = "Exceptionally beautiful skinny Korean girl, age 38, 174cm height. Very long wavy jet-black hair with thick bangs. Wearing blue contact lenses. Fair, flawless, porcelain-smooth sun-kissed skin."

DEFAULT_OUTFIT = "Wearing a very tiny light pink string bikini top with barely-there triangle cups, and a very tiny light pink G-string bikini bottom. The swimsuit covers almost nothing, showing off her entire body."

DEFAULT_POSE = "Lying flat on her back on a sunbed in a sexy, alluring pose, full body visible from head to feet. Her legs are slightly spread and one knee is gently bent to the side. One arm is raised above her head resting behind her, the other hand rests lightly on her stomach. Her back is arched slightly, emphasizing her body curves. She gazes directly upward at the camera with a sultry, inviting expression, lips slightly parted. Shot from directly above, perfectly vertical top-down bird's-eye view."

DEFAULT_SETTING = "Luxurious hotel rooftop swimming pool. White cushioned sunbed beside the sparkling blue pool. Potted tropical plants, elegant poolside furniture. Bright sunny day with clear blue sky."

DEFAULT_LIGHTING = "Bright midday sunlight. Warm sun-drenched highlights on her skin and the tiny pink swimsuit. Soft shadows on the white sunbed. Sparkling pool water reflections nearby."

DEFAULT_CAMERA = "Perfectly vertical overhead bird's-eye view, camera pointing straight down at 90 degrees, directly above the subject. Full body and sunbed entirely visible. Wide angle, tack-sharp focus, ultra-realistic 8K, professional fashion photography style."


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
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
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
    prompt,
    negative_prompt,
    width,
    height,
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

        # Run the pipeline
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=int(height),
            width=int(width),
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
        filename = (
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{width}x{height}"
            f"_gs{guidance_scale}_step{steps}_cfgnorm{cfg_normalization}_seed{int(seed)}.png"
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
        title="Z-Image Text-to-Image Generator",
    ) as interface:
        gr.Markdown("# Z-Image Text-to-Image Generator")
        gr.Markdown(
            f"Tongyi-MAI/Z-Image 모델을 사용하여 텍스트에서 이미지를 생성합니다."
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
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=768,
                        info="이미지 너비 (픽셀). 64의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
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
                        value=4.0,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. 권장: 4.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25,
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
                combined_prompt,
                prompt_negative,
                width,
                height,
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
