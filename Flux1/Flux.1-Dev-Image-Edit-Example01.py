import os

# Hub: Windows symlink 안내 숨김, 다운로드 타임아웃 완화 (import diffusers 전에 설정)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import gc
import platform
import re
import signal
import sys
import time
from datetime import datetime

import gradio as gr
import psutil
import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image


DEFAULT_IMAGE_PATH = "input02.jpg"
T5_MODEL_MAX_LENGTH = 512

SUBJECT = "A beautiful young korean woman."

FACE = ""

HEAD = "Move her face to make her see the viewer."

HEADWEAR = ""

BODY = ""

TOP = "She is wearing a dark blue bikini top."

ARM = ""

ARMWEAR = ""

HAND = ""

BOTTOM = "She is wearing a matching dark blue bikini bottom."

LEG = ""

LEGWEAR = ""

FOOT = ""

FOOTWEAR = ""

SETTING = ""

CAMERA = ""

LIGHTING = ""

POSITIVE = "8k resolution quality, high detail, high quality, best quality, realistic, masterpiece, cinematic lighting, photorealistic, sharp focus."

NEGATIVE = (
    "Blurry, out of focus, soft focus, hazy, low sharpness, grainy, low quality, "
    "deformed, bad anatomy, extra limbs, ugly, watermark, text, signature."
)


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def format_clip_prompt(subject: str) -> str:
    """CLIP 인코더용 문자열: Subject만 사용."""
    sections = [subject]
    return " ".join(normalize_spacing(s) for s in sections if s and s.strip())


def combine_t5_prompt_sections(
    subject,
    face,
    headwear,
    top,
    armwear,
    bottom,
    legwear,
    footwear,
    head,
    body,
    arm,
    hand,
    leg,
    foot,
    setting,
    camera,
    lighting,
    positive,
):
    """Combine separate prompt sections into the T5 (prompt_2) string.

    순서: Subject → 외모/의상(위→아래) → 포즈 → 보존할 배경/카메라/조명 → Positive
    """
    sections = [
        subject,
        face,
        headwear,
        top,
        armwear,
        bottom,
        legwear,
        footwear,
        head,
        body,
        arm,
        hand,
        leg,
        foot,
        setting,
        camera,
        lighting,
        positive,
    ]
    return " ".join(normalize_spacing(s) for s in sections if s and s.strip())


def combine_prompt_sections_dual(
    subject,
    face,
    headwear,
    top,
    armwear,
    bottom,
    legwear,
    footwear,
    head,
    body,
    arm,
    hand,
    leg,
    foot,
    setting,
    camera,
    lighting,
    positive,
):
    """Build CLIP and T5 prompt strings for FLUX.1 image edit dual encoders."""
    clip_prompt = format_clip_prompt(subject)
    t5_prompt = combine_t5_prompt_sections(
        subject, face, headwear, top, armwear, bottom, legwear, footwear, head,
        body, arm, hand, leg, foot, setting, camera, lighting, positive,
    )
    return clip_prompt, t5_prompt


DEFAULT_PROMPT_SECTIONS = (
    SUBJECT,
    FACE,
    HEADWEAR,
    TOP,
    ARMWEAR,
    BOTTOM,
    LEGWEAR,
    FOOTWEAR,
    HEAD,
    BODY,
    ARM,
    HAND,
    LEG,
    FOOT,
    SETTING,
    CAMERA,
    LIGHTING,
    POSITIVE,
)

DEFAULT_COMBINED_CLIP_PROMPT, DEFAULT_COMBINED_T5_PROMPT = combine_prompt_sections_dual(
    *DEFAULT_PROMPT_SECTIONS
)


def count_t5_tokens(text: str) -> int:
    """T5 토큰 수. 로드된 pipe.tokenizer_2 사용. 모델 미로드 시 단어 수 추정."""
    text = (text or "").strip()
    if not text:
        return 0
    tok = getattr(pipe, "tokenizer_2", None) if pipe is not None else None
    if tok is not None:
        ids = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
        return len(ids)
    return max(1, int(len(text.split()) * 1.2))


def format_t5_length_hint(t5_prompt: str, max_len: int = T5_MODEL_MAX_LENGTH) -> str:
    n = count_t5_tokens(t5_prompt)
    over = max(0, n - max_len)
    est = (
        ""
        if pipe is not None and getattr(pipe, "tokenizer_2", None) is not None
        else " (추정)"
    )
    if n == 0:
        return "T5: 비어 있음."
    if over > 0:
        return (
            f"T5 토큰 약 {n}{est} / 한도 {max_len} - 약 {over}토큰이 잘립니다. "
            "섹션을 짧게 하거나 T5 박스에서 직접 줄이세요."
        )
    return f"T5 토큰 약 {n}{est} / 한도 {max_len} (이 길이에서는 잘림 없음)."


def combine_prompt_sections_dual_with_stats(
    subject,
    face,
    headwear,
    top,
    armwear,
    bottom,
    legwear,
    footwear,
    head,
    body,
    arm,
    hand,
    leg,
    foot,
    setting,
    camera,
    lighting,
    positive,
):
    clip_prompt, t5_prompt = combine_prompt_sections_dual(
        subject, face, headwear, top, armwear, bottom, legwear, footwear, head,
        body, arm, hand, leg, foot, setting, camera, lighting, positive,
    )
    return clip_prompt, t5_prompt, format_t5_length_hint(t5_prompt)


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
    """Print compact hardware summary (CLI)."""
    print("=" * 60)
    print("하드웨어 사양(요약)")
    print("=" * 60)
    print(
        f"OS: {platform.system()} {platform.release()} ({platform.machine()}) | "
        f"Python: {platform.python_version()} | PyTorch: {torch.__version__}"
    )
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_physical} physical / {cpu_logical} logical cores")
    mem = psutil.virtual_memory()
    print(
        f"RAM: {mem.total / (1024**3):.1f} GB "
        f"(available {mem.available / (1024**3):.1f} GB)"
    )
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA: yes | GPUs: {gpu_count} | CUDA: {torch.version.cuda}")
        if gpu_count > 0:
            props0 = torch.cuda.get_device_properties(0)
            vram0 = props0.total_memory / (1024**3)
            print(f"GPU0: {props0.name} | VRAM: {vram0:.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"MPS: yes | device: {platform.processor()}")
    else:
        print("Device: CPU only")
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
    """Load and initialize the Flux image edit model with optimizations."""
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
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print("메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CUDA)")
    elif DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print("메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CPU)")
    elif DEVICE == "mps":
        pipe.enable_attention_slicing()
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)
        elif hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
        print("메모리 최적화 적용: attention slicing (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def load_default_image():
    """Load the default image when it exists."""
    if os.path.exists(DEFAULT_IMAGE_PATH):
        return Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
    return None


def resize_image_to_multiple(image: Image.Image, multiple: int = 16):
    """Resize image so width and height are divisible by the requested multiple."""
    width, height = image.size
    target_width = max(multiple, (width // multiple) * multiple)
    target_height = max(multiple, (height // multiple) * multiple)
    if (target_width, target_height) == (width, height):
        return image, False, (width, height), (target_width, target_height)

    resample = (
        Image.Resampling.LANCZOS
        if hasattr(Image, "Resampling")
        else Image.LANCZOS
    )
    resized = image.resize((target_width, target_height), resample=resample)
    return resized, True, (width, height), (target_width, target_height)


def generate_image(
    input_image,
    prompt_clip,
    prompt_t5,
    negative,
    strength,
    guidance_scale,
    true_cfg_scale,
    num_inference_steps,
    seed,
    max_sequence_length,
    image_format,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe

    if pipe is None:
        return None, None, "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요."

    try:
        if input_image is None:
            input_image = load_default_image()
            if input_image is None:
                return None, None, f"오류: 입력 이미지가 없고 {DEFAULT_IMAGE_PATH} 파일도 없습니다."

        prompt_clip = (prompt_clip or "").strip()
        if not prompt_clip:
            return None, None, "오류: CLIP 프롬프트가 비어 있습니다."
        prompt_t5 = (prompt_t5 or "").strip()
        if not prompt_t5:
            return None, None, "오류: T5 프롬프트(prompt_2)를 입력해주세요."
        negative = (negative or "").strip()

        input_image = input_image.convert("RGB")
        input_image, resized_input, original_size, adjusted_size = resize_image_to_multiple(
            input_image,
            multiple=16,
        )
        width, height = input_image.size
        steps = int(num_inference_steps)
        max_len = min(int(max_sequence_length), T5_MODEL_MAX_LENGTH)
        start_time = time.time()

        progress(0.0, desc="이미지 편집 준비 중...")
        print("이미지 편집 시작")
        print("=" * 60)
        print("[입력 CLIP 프롬프트]")
        print(prompt_clip)
        print("-" * 60)
        print("[입력 T5 프롬프트]")
        print(prompt_t5)
        if negative and float(true_cfg_scale) > 1.0:
            print("-" * 60)
            print("[네거티브 프롬프트]")
            print(negative)
        if resized_input:
            print(
                "입력 이미지 크기 보정: "
                f"{original_size[0]}x{original_size[1]} -> {adjusted_size[0]}x{adjusted_size[1]} "
                "(16px 배수)"
            )
        else:
            print(f"입력 이미지 크기: {width}x{height} (16px 배수)")
        print("=" * 60)

        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            progress(
                min(0.95, current / max(steps, 1) * 0.90 + 0.05),
                desc=f"편집 중... {current}/{steps}",
            )
            return callback_kwargs

        progress(0.05, desc="추론 시작...")
        pipe_kwargs = {
            "prompt": prompt_clip,
            "prompt_2": prompt_t5,
            "image": input_image,
            "width": width,
            "height": height,
            "strength": float(strength),
            "guidance_scale": float(guidance_scale),
            "true_cfg_scale": float(true_cfg_scale),
            "num_inference_steps": steps,
            "max_sequence_length": max_len,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }
        if negative and float(true_cfg_scale) > 1.0:
            pipe_kwargs["negative_prompt"] = negative
            pipe_kwargs["negative_prompt_2"] = negative

        with torch.inference_mode():
            output_image = pipe(**pipe_kwargs).images[0]

        progress(0.95, desc="이미지 저장 중...")
        elapsed = time.time() - start_time

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        if DEVICE == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            device_label = (
                gpu_name.replace(" ", "").replace("NVIDIA", "").replace("GeForce", "")
                + f"-{gpu_mem}GB"
            )
        else:
            device_label = DEVICE.upper()
        filename = (
            f"{script_name}_{timestamp}_{device_label}_{width}x{height}"
            f"_gs{guidance_scale}_step{steps}_seed{int(seed)}"
            f"_cfg{true_cfg_scale}_str{strength}_msl{max_len}.{ext}"
        )

        if image_format == "JPEG":
            output_image.save(filename, format="JPEG", quality=100, subsampling=0)
        else:
            output_image.save(filename)

        resize_info = (
            f" | 크기 보정: {original_size[0]}x{original_size[1]} -> {width}x{height}"
            if resized_input
            else ""
        )
        print(f"이미지 편집 완료! 소요 시간: {elapsed:.1f}초 | 저장됨: {filename}")
        progress(1.0, desc="완료!")
        return output_image, filename, f"✓ 완료! ({elapsed:.1f}초){resize_info} | 저장됨: {filename}"
    except Exception as e:
        return None, None, f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()
    load_model()

    with gr.Blocks(title="Flux.1-dev Image Edit Generator") as interface:
        gr.Markdown("# Flux.1-dev Image Edit Generator")
        gr.Markdown(
            f"AI를 사용하여 입력 이미지를 프롬프트 방향으로 편집합니다."
            f" (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
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

                gr.Markdown("### 입력 이미지")
                default_image = load_default_image()
                input_image = gr.Image(
                    type="pil",
                    label="입력 이미지",
                    value=default_image,
                    height=700,
                )

                gr.Markdown(
                    "### 프롬프트 구성\n"
                    "- **CLIP**: Subject만 사용합니다.\n"
                    "- **T5**: Subject → 얼굴/외모 → 의상(위→아래) → 포즈 → 보존할 배경/카메라/조명 → Positive 순.\n"
                    f"- **T5 한도**: FLUX.1-dev는 최대 **{T5_MODEL_MAX_LENGTH}토큰**(초과분 잘림).\n"
                    "- **Negative**: True CFG > 1.0일 때 적용됩니다."
                )
                with gr.Accordion("프롬프트 섹션 (Subject · 외모 · 의상 · 포즈 · 보존 설정 · Positive)", open=True):
                    prompt_subject = gr.Textbox(
                        label="1. 주제/대상 (Subject)",
                        value=SUBJECT,
                        lines=2,
                        placeholder="예: a young woman, a portrait photo, a product photo",
                        info="CLIP에는 Subject만 들어갑니다.",
                    )
                    prompt_face = gr.Textbox(
                        label="2. 얼굴/외모 (Face)",
                        value=FACE,
                        lines=2,
                        placeholder="예: preserve the same face, natural skin texture",
                        info="얼굴, 피부, 눈, 표정, 머리카락 및 보존 조건을 설명합니다.",
                    )
                    prompt_headwear = gr.Textbox(
                        label="3. 머리 장식 (Headwear)",
                        value=HEADWEAR,
                        lines=1,
                        placeholder="예: black beret, no hat, preserve original hair accessory",
                        info="모자, 헤어핀, 머리띠 등 머리 장식을 설명합니다.",
                    )
                    prompt_top = gr.Textbox(
                        label="4. 상의 (Top)",
                        value=TOP,
                        lines=2,
                        placeholder="예: wearing a dark blue bikini top",
                        info="상의, 속옷 상의, 변경할 옷을 설명합니다.",
                    )
                    prompt_armwear = gr.Textbox(
                        label="5. 팔 장식 (Armwear)",
                        value=ARMWEAR,
                        lines=1,
                        placeholder="예: preserve original bracelet, black lace gloves",
                        info="장갑, 팔찌, 소매 장식 등을 설명합니다.",
                    )
                    prompt_bottom = gr.Textbox(
                        label="6. 하의 (Bottom)",
                        value=BOTTOM,
                        lines=2,
                        placeholder="예: matching dark blue bikini bottom",
                        info="하의, 속옷 하의, 변경할 옷을 설명합니다.",
                    )
                    prompt_legwear = gr.Textbox(
                        label="7. 레그웨어 (Legwear)",
                        value=LEGWEAR,
                        lines=1,
                        placeholder="예: preserve original stockings, bare legs",
                        info="스타킹, 양말, 레깅스 등을 설명합니다.",
                    )
                    prompt_footwear = gr.Textbox(
                        label="8. 신발 (Footwear)",
                        value=FOOTWEAR,
                        lines=1,
                        placeholder="예: preserve original shoes, white sandals",
                        info="신발, 부츠, 샌들 등을 설명합니다.",
                    )
                    prompt_head = gr.Textbox(
                        label="9. 포즈 - 머리 (Head)",
                        value=HEAD,
                        lines=1,
                        placeholder="예: head turned toward the camera",
                        info="머리와 시선 방향을 설명합니다.",
                    )
                    prompt_body = gr.Textbox(
                        label="10. 포즈 - 몸통 (Body)",
                        value=BODY,
                        lines=2,
                        placeholder="예: preserve the original body pose",
                        info="몸통 자세와 전체 실루엣을 설명합니다.",
                    )
                    prompt_arm = gr.Textbox(
                        label="11. 포즈 - 팔 (Arm)",
                        value=ARM,
                        lines=1,
                        placeholder="예: preserve the original arm position",
                        info="팔의 위치와 자세를 설명합니다.",
                    )
                    prompt_hand = gr.Textbox(
                        label="12. 포즈 - 손 (Hand)",
                        value=HAND,
                        lines=1,
                        placeholder="예: preserve natural fingers",
                        info="손의 위치와 동작을 설명합니다.",
                    )
                    prompt_leg = gr.Textbox(
                        label="13. 포즈 - 다리 (Leg)",
                        value=LEG,
                        lines=1,
                        placeholder="예: preserve the original leg position",
                        info="다리 자세를 설명합니다.",
                    )
                    prompt_foot = gr.Textbox(
                        label="14. 포즈 - 발 (Foot)",
                        value=FOOT,
                        lines=1,
                        placeholder="예: preserve the original foot position",
                        info="발의 위치를 설명합니다.",
                    )
                    prompt_setting = gr.Textbox(
                        label="15. 배경/장소 보존 (Setting)",
                        value=SETTING,
                        lines=2,
                        placeholder="예: preserve the original background",
                        info="배경, 장소, 환경, 계절 및 보존 조건을 설명합니다.",
                    )
                    prompt_camera = gr.Textbox(
                        label="16. 카메라 설정 보존 (Camera)",
                        value=CAMERA,
                        lines=2,
                        placeholder="예: preserve the same camera angle and crop",
                        info="카메라 각도, 렌즈감, 프레이밍, 크롭을 설명합니다.",
                    )
                    prompt_lighting = gr.Textbox(
                        label="17. 조명 보존 (Lighting)",
                        value=LIGHTING,
                        lines=2,
                        placeholder="예: preserve the original lighting and shadows",
                        info="조명 조건, 빛의 방향, 색온도, 노출을 설명합니다.",
                    )
                    positive_box = gr.Textbox(
                        label="18. 포지티브 프롬프트 (Positive)",
                        value=POSITIVE,
                        lines=2,
                        placeholder="예: photorealistic, high detail, sharp focus",
                        info="T5 프롬프트 맨 뒤에 포함됩니다.",
                    )
                with gr.Accordion("네거티브 프롬프트 (Negative)", open=False):
                    negative_box = gr.Textbox(
                        label="네거티브 프롬프트",
                        value=NEGATIVE,
                        lines=3,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="True CFG Scale > 1.0일 때 적용됩니다.",
                    )
                with gr.Accordion("CLIP 프롬프트 (Subject만)", open=False):
                    prompt_clip = gr.Textbox(
                        label="CLIP 프롬프트",
                        value=DEFAULT_COMBINED_CLIP_PROMPT,
                        lines=3,
                        interactive=True,
                        info="Subject만 자동 반영됩니다.",
                    )
                with gr.Accordion("T5 프롬프트 (전체 섹션 결합)", open=False):
                    prompt_t5 = gr.Textbox(
                        label="T5 프롬프트",
                        value=DEFAULT_COMBINED_T5_PROMPT,
                        lines=5,
                        interactive=True,
                        info="섹션들이 자동 결합됩니다. 필요하면 직접 수정할 수 있습니다.",
                    )
                    t5_length_hint = gr.Textbox(
                        label="T5 길이",
                        value=format_t5_length_hint(DEFAULT_COMBINED_T5_PROMPT),
                        lines=2,
                        interactive=False,
                    )
                prompt_sections = [
                    prompt_subject,
                    prompt_face,
                    prompt_headwear,
                    prompt_top,
                    prompt_armwear,
                    prompt_bottom,
                    prompt_legwear,
                    prompt_footwear,
                    prompt_head,
                    prompt_body,
                    prompt_arm,
                    prompt_hand,
                    prompt_leg,
                    prompt_foot,
                    prompt_setting,
                    prompt_camera,
                    prompt_lighting,
                    positive_box,
                ]
                for section in prompt_sections:
                    section.change(
                        fn=combine_prompt_sections_dual_with_stats,
                        inputs=prompt_sections,
                        outputs=[prompt_clip, prompt_t5, t5_length_hint],
                    )
                prompt_t5.change(
                    fn=format_t5_length_hint,
                    inputs=[prompt_t5],
                    outputs=[t5_length_hint],
                )

            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    strength = gr.Slider(
                        label="Strength (변경 강도)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.85,
                        step=0.05,
                        info="낮을수록 원본 유지, 높을수록 변화가 커집니다.",
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=20.0,
                        value=3.5,
                        step=0.5,
                        info="프롬프트 준수도. Flux.1 Dev 권장: 3.5",
                    )

                with gr.Row():
                    true_cfg_scale = gr.Slider(
                        label="True CFG Scale (네거티브 프롬프트 강도)",
                        minimum=1.0,
                        maximum=5.0,
                        value=1.5,
                        step=0.5,
                        info="1.0이면 네거티브 프롬프트 비활성화. 1.5~2.0 권장.",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=50,
                        value=28,
                        step=1,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가.",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=100,
                        precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    max_sequence_length = gr.Slider(
                        label="최대 시퀀스 길이",
                        minimum=64,
                        maximum=512,
                        value=512,
                        step=64,
                        info="텍스트 인코더 최대 길이. 긴 프롬프트는 높은 값 필요.",
                    )

                image_format = gr.Radio(
                    label="이미지 포맷",
                    choices=["JPEG", "PNG"],
                    value="PNG",
                    info="JPEG: quality 100 (4:4:4), PNG: 무손실 압축.",
                )

                gr.Markdown("---")
                gr.Markdown("### 이미지 편집")
                generate_btn = gr.Button("이미지 편집", variant="primary", size="lg")
                output_image = gr.Image(label="편집된 이미지", type="pil", height=700)
                output_file = gr.File(label="파일 다운로드")
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

        generate_btn.click(
            fn=generate_image,
            inputs=[
                input_image,
                prompt_clip,
                prompt_t5,
                negative_box,
                strength,
                guidance_scale,
                true_cfg_scale,
                num_inference_steps,
                seed,
                max_sequence_length,
                image_format,
            ],
            outputs=[output_image, output_file, output_message],
        )

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
