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

DEFAULT_IMAGE = "Test02.jpg"

# Default values for each prompt section
DEFAULT_QUALITY = ""
DEFAULT_NEGATIVE = "Perfect anatomy, missing fingers, no extra fingers, no malformed fingers, no fused fingers. no missing toes, no extra toes, no deformed toes, no fused toes."
DEFAULT_APPEARANCE = ""
DEFAULT_OUTFIT = "She wears very small minimal pink bikini bottom."
DEFAULT_POSE = "Her one hand touching her hair softly, the other hand just below of the elbow of the arm"
DEFAULT_SETTING = ""
DEFAULT_LIGHTING = ""
DEFAULT_CAMERA = ""
DEFAULT_NEGATIVE_PROMPT = ""


def combine_prompt_sections(
    quality, negative, appearance, outfit, pose, setting, lighting, camera
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [quality, negative, appearance, outfit, pose, setting, lighting, camera]
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
    """입력 이미지가 업로드되면 높이/너비 슬라이더를 이미지 크기로 업데이트."""
    if image is None:
        return gr.update(), gr.update()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    w, h = image.size
    return gr.update(value=h), gr.update(value=w)


def generate_image(
    input_image,
    prompt,
    negative_prompt,
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
            progress_val = 0.05 + ratio * 0.85
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

        # Build final prompt (merge negative prompt for models that don't support the param)
        final_prompt = prompt
        if negative_prompt and negative_prompt.strip():
            final_prompt = f"{prompt}, avoid: {negative_prompt.strip()}"

        pipe_kwargs = {
            "prompt": final_prompt,
            "image": input_image,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        if negative_prompt and negative_prompt.strip():
            try:
                test_kwargs = pipe_kwargs.copy()
                test_kwargs["negative_prompt"] = negative_prompt.strip()
                image = pipe(**test_kwargs).images[0]
                print("부정 프롬프트 파라미터가 적용되었습니다.")
            except TypeError as e:
                if "negative_prompt" in str(e):
                    print("이 모델은 negative_prompt 파라미터를 지원하지 않습니다. 메인 프롬프트에 포함시켜 생성합니다.")
                    image = pipe(**pipe_kwargs).images[0]
                else:
                    raise e
        else:
            image = pipe(**pipe_kwargs).images[0]

        progress(0.95, desc="이미지 저장 중...")

        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        filename = (
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{width}x{height}"
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

    print_hardware_info()

    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    with gr.Blocks(
        title="Flux.2 Klein 9B Image-to-Image Generator",
    ) as interface:
        gr.Markdown("# Flux.2 Klein 9B Image-to-Image Generator")
        gr.Markdown(
            f"이미지를 업로드하고 프롬프트로 편집하세요."
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
                    label="2. 해부학/제약 (Anatomy & Constraints)",
                    value=DEFAULT_NEGATIVE,
                    lines=2,
                    placeholder="예: perfect anatomy, no extra fingers",
                    info="해부학적 정확성 및 생성 제약 조건을 지정합니다.",
                )
                prompt_appearance = gr.Textbox(
                    label="3. 외모 (Appearance)",
                    value=DEFAULT_APPEARANCE,
                    lines=2,
                    placeholder="예: A beautiful Korean girl with long black hair",
                    info="인물의 외모, 얼굴, 머리카락, 나이 등을 설명합니다.",
                )
                prompt_outfit = gr.Textbox(
                    label="4. 의상 (Outfit)",
                    value=DEFAULT_OUTFIT,
                    lines=2,
                    placeholder="예: in a red bikini, wearing a white dress",
                    info="의상, 액세서리, 착용한 아이템을 설명합니다.",
                )
                prompt_pose = gr.Textbox(
                    label="5. 포즈/구도 (Pose & Composition)",
                    value=DEFAULT_POSE,
                    lines=2,
                    placeholder="예: standing, looking over shoulder, full body",
                    info="자세, 시선 방향, 카메라 앵글, 촬영 구도를 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="6. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    placeholder="예: on a boardwalk at sunset, calm ocean",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="7. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    placeholder="예: golden hour, soft glow, cinematic lighting",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="8. 카메라 설정 (Camera Settings)",
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
                            DEFAULT_NEGATIVE,
                            DEFAULT_APPEARANCE,
                            DEFAULT_OUTFIT,
                            DEFAULT_POSE,
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
                    prompt_negative,
                    prompt_appearance,
                    prompt_outfit,
                    prompt_pose,
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

                negative_prompt_input = gr.Textbox(
                    label="부정 프롬프트 (Negative Prompt)",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=2,
                    placeholder="예: blurry, bad quality, distorted",
                    info="이미지에 포함되지 않기를 원하는 요소 (모델이 지원하는 경우에만 적용됨).",
                )

            # Right column: Input image + Parameters + Image generation
            with gr.Column(scale=1):
                gr.Markdown("### 입력 이미지")
                input_image = gr.Image(
                    label="입력 이미지",
                    type="pil",
                    value=DEFAULT_IMAGE,
                    height=800,
                )

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
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. 권장: 0.5-1.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=4,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 4-12",
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
                generate_btn = gr.Button("이미지 편집", variant="primary", size="lg")
                output_image = gr.Image(label="생성된 이미지", height=700)
                output_message = gr.Textbox(label="상태", interactive=False)

        load_model_btn.click(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        )

        device_selector.change(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        )

        input_image.change(
            fn=on_image_upload,
            inputs=[input_image],
            outputs=[height, width],
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                input_image,
                combined_prompt,
                negative_prompt_input,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                image_format,
            ],
            outputs=[output_image, output_message],
        )

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
