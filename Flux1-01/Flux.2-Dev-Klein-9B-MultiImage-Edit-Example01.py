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

DEFAULT_BASE_IMAGE = ""
DEFAULT_REF_IMAGE1 = ""
DEFAULT_REF_IMAGE2 = ""
DEFAULT_REF_IMAGE3 = ""
DEFAULT_REF_IMAGE4 = ""

# Default values for each prompt section
DEFAULT_SUBJECT = ""
DEFAULT_FACE = ""
DEFAULT_POSE_HEAD = ""
DEFAULT_POSE_BODY = ""
DEFAULT_POSE_ARM = "Bare arms."
DEFAULT_POSE_HAND = ""
DEFAULT_POSE_LEG = ""
DEFAULT_POSE_FOOT = ""
DEFAULT_HEADWEAR = ""
DEFAULT_TOP = "She is wearing a white underboob, 'I Love Seoul' written on it."
DEFAULT_BOTTOM = ""
DEFAULT_LEGWEAR = ""
DEFAULT_FOOTWEAR = ""
DEFAULT_ARMWEAR = ""
DEFAULT_SETTING = ""
DEFAULT_LIGHTING = ""
DEFAULT_CAMERA = ""


def load_default_image():
    """Load the default base image if it exists."""
    if not DEFAULT_BASE_IMAGE:
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, DEFAULT_BASE_IMAGE)
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    return None


def load_default_ref_image(filename):
    """Load a default reference image if it exists."""
    if not filename:
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, filename)
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    return None


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    subject, face,
    pose_head, pose_body, pose_arm, pose_hand, pose_leg, pose_foot,
    headwear, top, bottom, legwear, footwear, armwear,
    setting, lighting, camera
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [
        subject, face,
        pose_head, pose_body, pose_arm, pose_hand, pose_leg, pose_foot,
        headwear, top, bottom, legwear, footwear, armwear,
        setting, lighting, camera,
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
        print("메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing")
    elif DEVICE == "mps":
        pipe.to(DEVICE)
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def on_base_image_upload(image):
    """기본 이미지 업로드 시 출력 크기 자동 설정."""
    if image is None:
        return 768, 1536, "이미지를 업로드하면 원본 크기가 표시됩니다."
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    w, h = image.size
    rw, rh = round_to_64(w), round_to_64(h)
    info = f"기본 이미지: {w} × {h} px  →  출력 크기: {rw} × {rh} px (64 배수로 반올림)"
    return rw, rh, info


def generate_image(
    base_image,
    ref_image1,
    ref_image2,
    ref_image3,
    ref_image4,
    prompt_subject,
    prompt_face,
    prompt_pose_head,
    prompt_pose_body,
    prompt_pose_arm,
    prompt_pose_hand,
    prompt_pose_leg,
    prompt_pose_foot,
    prompt_headwear,
    prompt_top,
    prompt_bottom,
    prompt_legwear,
    prompt_footwear,
    prompt_armwear,
    prompt_setting,
    prompt_lighting,
    prompt_camera,
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
        return None, "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요."

    # --- 기본 이미지 준비 ---
    if base_image is None:
        base_image = load_default_image()
    if base_image is None:
        return None, "오류: 기본 이미지를 업로드해주세요."
    if not isinstance(base_image, Image.Image):
        base_image = Image.fromarray(base_image)

    # --- 참조 이미지 준비 ---
    ref_images = []
    for ref in [ref_image1, ref_image2, ref_image3, ref_image4]:
        if ref is not None:
            ref_images.append(ref if isinstance(ref, Image.Image) else Image.fromarray(ref))

    if not ref_images:
        return None, "오류: 참조 이미지를 최소 1개 이상 업로드해주세요."

    # --- 프롬프트 섹션 합치기 ---
    prompt = combine_prompt_sections(
        prompt_subject, prompt_face,
        prompt_pose_head, prompt_pose_body, prompt_pose_arm, prompt_pose_hand,
        prompt_pose_leg, prompt_pose_foot,
        prompt_headwear, prompt_top, prompt_bottom, prompt_legwear, prompt_footwear, prompt_armwear,
        prompt_setting, prompt_lighting, prompt_camera,
    )
    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        out_w, out_h = int(width), int(height)
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="이미지 준비 중...")
        print(f"이미지 준비: 기본 1개 + 참조 {len(ref_images)}개")

        # --- 모델에 전달할 이미지 리스트 구성 ---
        input_images = [base_image.resize((out_w, out_h), Image.LANCZOS).convert("RGB")]
        for ref in ref_images:
            input_images.append(ref.resize((out_w, out_h), Image.LANCZOS).convert("RGB"))
        print(f"총 {len(input_images)}개 이미지를 파이프라인에 리스트로 전달")

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
            result = pipe(
                prompt=prompt,
                image=input_images,
                width=out_w,
                height=out_h,
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
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{out_w}x{out_h}"
            f"_gs{guidance_scale}_step{steps}_seed{int(seed)}.{ext}"
        )

        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")
        print(f"이미지가 저장되었습니다 : {filename}")
        if image_format == "JPEG":
            result.save(filename, format="JPEG", quality=100, subsampling=0)
        else:
            result.save(filename)

        progress(1.0, desc="완료!")
        return (
            result,
            f"✓ 완료! ({elapsed:.1f}초) | 입력: 기본 1개 + 참조 {len(ref_images)}개 | 저장됨: {filename}",
        )

    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()

    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    _default_img = load_default_image()
    _default_ref1 = load_default_ref_image(DEFAULT_REF_IMAGE1)
    _default_ref2 = load_default_ref_image(DEFAULT_REF_IMAGE2)
    _default_ref3 = load_default_ref_image(DEFAULT_REF_IMAGE3)
    _default_ref4 = load_default_ref_image(DEFAULT_REF_IMAGE4)
    _init_w, _init_h = get_image_dimensions(_default_img)
    _default_info = (
        f"기본 이미지: {DEFAULT_BASE_IMAGE} ({_default_img.size[0]} × {_default_img.size[1]} px)"
        if _default_img is not None
        else "이미지를 업로드하면 원본 크기가 표시됩니다."
    )

    with gr.Blocks(title="Flux.2 Klein 9B Multi-Image Compositor") as interface:
        gr.Markdown("# Flux.2 Klein 9B Multi-Image Compositor")
        gr.Markdown(
            f"기본 이미지와 여러 참조 이미지를 합성하여 하나의 이미지를 생성합니다."
            f" (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            # ── 왼쪽 컬럼: 모델 설정 + 이미지 입력 + 프롬프트 ──
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
                        else "모델이 로드되지 않았습니다. 디바이스를 선택하고 '모델 로드' 버튼을 눌러주세요."
                    ),
                    interactive=False,
                )

                gr.Markdown("### 기본 이미지 (Base Image)")
                base_image = gr.Image(
                    label="기본 이미지 (합성의 베이스)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=300,
                    value=_default_img,
                )
                image_info = gr.Textbox(
                    label="이미지 정보",
                    value=_default_info,
                    interactive=False,
                )

                gr.Markdown("### 참조 이미지들 (Reference Images) - 최대 4개")
                with gr.Row():
                    ref_image1 = gr.Image(
                        label="참조 이미지 1",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200,
                        value=_default_ref1,
                    )
                    ref_image2 = gr.Image(
                        label="참조 이미지 2",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200,
                        value=_default_ref2,
                    )
                with gr.Row():
                    ref_image3 = gr.Image(
                        label="참조 이미지 3",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200,
                        value=_default_ref3,
                    )
                    ref_image4 = gr.Image(
                        label="참조 이미지 4",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=200,
                        value=_default_ref4,
                    )

                gr.Markdown("### 프롬프트 구성")
                prompt_subject = gr.Textbox(
                    label="1. 주제/대상 (Subject)",
                    value=DEFAULT_SUBJECT,
                    lines=2,
                    placeholder="예: 1girl, young woman, a cat",
                    info="이미지의 주된 주제나 대상을 설명합니다.",
                )
                prompt_face = gr.Textbox(
                    label="2. 얼굴/외모 (Face)",
                    value=DEFAULT_FACE,
                    lines=2,
                    placeholder="예: fair complexion, blue contact lenses, soft smile",
                    info="얼굴, 피부, 눈, 표정, 머리카락 등을 설명합니다.",
                )
                prompt_pose_head = gr.Textbox(
                    label="3a. 포즈 - 머리 (Pose: Head)",
                    value=DEFAULT_POSE_HEAD,
                    lines=1,
                    placeholder="예: head tilted slightly, gazing off-camera",
                    info="머리와 시선 방향을 설명합니다.",
                )
                prompt_pose_body = gr.Textbox(
                    label="3b. 포즈 - 몸통 (Pose: Body)",
                    value=DEFAULT_POSE_BODY,
                    lines=2,
                    placeholder="예: body angled slightly, leaning forward",
                    info="몸통 자세와 전체 실루엣을 설명합니다.",
                )
                prompt_pose_arm = gr.Textbox(
                    label="3c. 포즈 - 팔 (Pose: Arm)",
                    value=DEFAULT_POSE_ARM,
                    lines=1,
                    placeholder="예: arms resting across torso",
                    info="팔의 위치와 자세를 설명합니다.",
                )
                prompt_pose_hand = gr.Textbox(
                    label="3d. 포즈 - 손 (Pose: Hand)",
                    value=DEFAULT_POSE_HAND,
                    lines=1,
                    placeholder="예: one hand gripping the other arm",
                    info="손의 위치와 동작을 설명합니다.",
                )
                prompt_pose_leg = gr.Textbox(
                    label="3e. 포즈 - 다리 (Pose: Leg)",
                    value=DEFAULT_POSE_LEG,
                    lines=1,
                    placeholder="예: one leg stepping forward, weight on left leg",
                    info="다리 자세를 설명합니다.",
                )
                prompt_pose_foot = gr.Textbox(
                    label="3f. 포즈 - 발 (Pose: Foot)",
                    value=DEFAULT_POSE_FOOT,
                    lines=1,
                    placeholder="예: feet slightly apart, toes pointed forward",
                    info="발의 위치를 설명합니다.",
                )
                prompt_headwear = gr.Textbox(
                    label="4. 머리 장식 (Headwear)",
                    value=DEFAULT_HEADWEAR,
                    lines=1,
                    placeholder="예: black beret, floral hairpin",
                    info="모자, 헤어핀, 머리띠 등 머리 장식을 설명합니다.",
                )
                prompt_top = gr.Textbox(
                    label="5. 상의 (Top)",
                    value=DEFAULT_TOP,
                    lines=2,
                    placeholder="예: sheer black button-up shirt, tiny black bra",
                    info="상의, 속옷 상의 등을 설명합니다.",
                )
                prompt_bottom = gr.Textbox(
                    label="6. 하의 (Bottom)",
                    value=DEFAULT_BOTTOM,
                    lines=1,
                    placeholder="예: tiny black panty, mini skirt",
                    info="하의, 속옷 하의 등을 설명합니다.",
                )
                prompt_legwear = gr.Textbox(
                    label="7. 레그웨어 (Legwear)",
                    value=DEFAULT_LEGWEAR,
                    lines=1,
                    placeholder="예: thigh-high black stockings, sheer tights",
                    info="스타킹, 양말, 레깅스 등을 설명합니다.",
                )
                prompt_footwear = gr.Textbox(
                    label="8. 신발 (Footwear)",
                    value=DEFAULT_FOOTWEAR,
                    lines=1,
                    placeholder="예: black stiletto heels, white sneakers",
                    info="신발, 부츠, 샌들 등을 설명합니다.",
                )
                prompt_armwear = gr.Textbox(
                    label="9. 팔 장식 (Armwear)",
                    value=DEFAULT_ARMWEAR,
                    lines=1,
                    placeholder="예: black lace gloves, silver bracelet",
                    info="장갑, 팔찌, 소매 장식 등을 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="10. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    placeholder="예: rooftop terrace at twilight, city skyline",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="11. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    placeholder="예: golden hour, city glow, cinematic rim light",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="12. 카메라 설정 (Camera Settings)",
                    value=DEFAULT_CAMERA,
                    lines=2,
                    placeholder="예: Sony A7R V, 85mm f/1.8, ISO 400",
                    info="카메라 기종, 렌즈, ISO, 셔터 스피드, 조리개, 피사계 심도 등을 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            DEFAULT_SUBJECT, DEFAULT_FACE,
                            DEFAULT_POSE_HEAD, DEFAULT_POSE_BODY, DEFAULT_POSE_ARM, DEFAULT_POSE_HAND,
                            DEFAULT_POSE_LEG, DEFAULT_POSE_FOOT,
                            DEFAULT_HEADWEAR, DEFAULT_TOP, DEFAULT_BOTTOM,
                            DEFAULT_LEGWEAR, DEFAULT_FOOTWEAR, DEFAULT_ARMWEAR,
                            DEFAULT_SETTING, DEFAULT_LIGHTING, DEFAULT_CAMERA,
                        ),
                        lines=4,
                        interactive=False,
                        info="위 섹션들이 자동으로 합쳐진 최종 프롬프트입니다.",
                    )
                prompt_sections = [
                    prompt_subject, prompt_face,
                    prompt_pose_head, prompt_pose_body, prompt_pose_arm, prompt_pose_hand,
                    prompt_pose_leg, prompt_pose_foot,
                    prompt_headwear, prompt_top, prompt_bottom,
                    prompt_legwear, prompt_footwear, prompt_armwear,
                    prompt_setting, prompt_lighting, prompt_camera,
                ]
                for section in prompt_sections:
                    section.change(
                        fn=combine_prompt_sections,
                        inputs=prompt_sections,
                        outputs=[combined_prompt],
                    )

            # ── 오른쪽 컬럼: 파라미터 + 생성 ──
            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="출력 너비",
                        minimum=256, maximum=2048, step=64,
                        value=_init_w,
                        info="출력 이미지 너비 (픽셀).",
                    )
                    height = gr.Slider(
                        label="출력 높이",
                        minimum=256, maximum=2048, step=64,
                        value=_init_h,
                        info="출력 이미지 높이 (픽셀).",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.0, maximum=1.0, step=0.05, value=1.0,
                        info="Klein 권장: 1.0. 낮으면 창의적, 높으면 정확.",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1, maximum=20, step=1, value=4,
                        info="Klein 권장: 4. 권장 범위: 4-12",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드", value=42, precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    image_format = gr.Radio(
                        label="이미지 포맷",
                        choices=["JPEG", "PNG"], value="JPEG",
                        info="JPEG: quality 100, PNG: 무손실.",
                    )

                gr.Markdown("### 이미지 생성")
                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        # ── 이벤트 연결 ──
        base_image.change(
            fn=on_base_image_upload,
            inputs=[base_image],
            outputs=[width, height, image_info],
        )

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

        # 이미지 생성 버튼
        generate_btn.click(
            fn=generate_image,
            inputs=[
                base_image, ref_image1, ref_image2, ref_image3, ref_image4,
                prompt_subject, prompt_face,
                prompt_pose_head, prompt_pose_body, prompt_pose_arm, prompt_pose_hand,
                prompt_pose_leg, prompt_pose_foot,
                prompt_headwear, prompt_top, prompt_bottom,
                prompt_legwear, prompt_footwear, prompt_armwear,
                prompt_setting, prompt_lighting, prompt_camera,
                width, height,
                guidance_scale, num_inference_steps,
                seed, image_format,
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
