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

# Default values for each prompt section
SUBJECT = "A full body photography of a beautiful young skinny Korean woman with soft idol aesthetics standing on a casual spring outing in Seoul."

FOOT = "Both feet pressed firmly together with ankles touching, inner sides of both feet in contact, toes pointing straight forward, white sneakers clearly visible side by side, feet entirely in frame and not cropped."

LEG = "Both legs straight and pressed firmly together, knees touching, thighs together, no gap between legs."

FACE = "She has a fair, clear complexion. Her expression is innocent and curious in the eyes, with a faint soft smile and lips gently closed, looking directly at the camera. She has long wavy voluminous jet-black hair with beautiful soft waves and curls, hair entirely on the back."

BODY = "Standing perfectly still and upright, both feet and legs together, body facing completely straight toward the camera, chest and torso fully frontal, posture tall and elegant, shoulders back."

ARM = "Both arms hanging naturally and relaxed at her sides."

HAND = "Both hands hanging gracefully at her sides with fingers lightly extended."

FOOTWEAR = "Clean white canvas sneakers, simple and casual."

LEGWEAR = "Bare legs, smooth and fair skin visible through the skirt slit, legs themselves pressed together."

BOTTOM = ""

TOP = "Long dark navy chiffon long one-piece dress with thin spaghetti straps, simple neckline, bare shoulders and arms, mostly opaque with only a slight translucency, densely scattered tiny cherry blossom print in soft pink and white, fitted waist, flowing A-line skirt with a side slit slightly opened by a very calm light breeze, skirt hem and fabric lifted only a little on one side, subtle flutter, revealing the bare leg while both legs remain firmly pressed together."

HEADWEAR = ""

ARMWEAR = ""

HEAD = "Head tilted slightly to one side, gentle and relaxed posture."

SETTING = "Bright spring street in Seoul, cherry blossom trees lining the sidewalk with pink petals falling gently, warm sunny day, clean pavement."

LIGHTING = "Bright even spring daylight, soft frontal natural light, face clearly and brightly lit, no harsh shadows."

CAMERA = "Full body shot, entire body from head to feet fully in frame, feet and sneakers not cropped, waist level angle, subject facing camera, sharp focus, soft bokeh background."

POSITIVE = "8k, high quality, realistic, detailed, sharp focus, perfect anatomy, ten fingers."

MAX_SEQUENCE_LENGTH = 512

# Default base image path (None or file path string, e.g. r"C:\Users\user\Desktop\test.jpg")
DEFAULT_IMAGE_PATH = "Test01.jpg"


def make_image_grid(images: list) -> Image.Image:
    """Arrange PIL images into a grid that fits in one view."""
    n = len(images)
    if n == 1:
        return images[0]
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), color=(20, 20, 20))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(img, (c * w, r * h))
    return grid


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    subject,
    face,
    pose_head,
    pose_body,
    pose_arm,
    pose_hand,
    pose_leg,
    pose_foot,
    headwear,
    top,
    bottom,
    legwear,
    footwear,
    armwear,
    setting,
    lighting,
    camera,
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [
        subject,
        face,
        pose_head,
        pose_body,
        pose_arm,
        pose_hand,
        pose_leg,
        pose_foot,
        headwear,
        top,
        bottom,
        legwear,
        footwear,
        armwear,
        setting,
        lighting,
        camera,
    ]
    combined = ", ".join(normalize_spacing(s) for s in sections if s and s.strip())
    return combined


def round_to_16(value: int) -> int:
    """Round value to the nearest multiple of 16, minimum 256."""
    return max(256, round(value / 16) * 16)


def get_image_dimensions(image):
    """Read uploaded image size and return width/height rounded to 16."""
    if image is None:
        return 768, 1536
    w, h = image.size
    return round_to_16(w), round_to_16(h)


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
        f"RAM: {mem.total / (1024**3):.1f} GB (available {mem.available / (1024**3):.1f} GB)"
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


def on_base_image_upload(image):
    """기본 이미지 업로드 시 출력 크기 자동 설정."""
    if image is None:
        return 768, 1536, "이미지를 업로드하면 원본 크기가 표시됩니다."
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    w, h = image.size
    rw, rh = round_to_16(w), round_to_16(h)
    info = f"기본 이미지: {w} × {h} px  →  출력 크기: {rw} × {rh} px (16 배수로 반올림)"
    return rw, rh, info


def generate_image(
    base_image,
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
    num_inference_steps,
    num_images_per_prompt,
    seed,
    positive_prompt,
    max_sequence_length,
    image_format,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe

    if pipe is None:
        return (
            None,
            [],
            [],
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    # --- 기본 이미지 준비 (선택사항) ---
    if base_image is not None and not isinstance(base_image, Image.Image):
        base_image = Image.fromarray(base_image)


    # --- 프롬프트 섹션 합치기 ---
    prompt = combine_prompt_sections(
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
    )
    if not prompt:
        return None, [], [], "오류: 프롬프트를 입력해주세요."

    # Append positive prompt
    if positive_prompt and positive_prompt.strip():
        prompt = prompt.rstrip() + " " + positive_prompt.strip()

    try:
        out_w, out_h = int(width), int(height)
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="프롬프트 준비 중...")
        print("=" * 60)
        print("[입력 프롬프트]")
        print(prompt)
        print("=" * 60)

        # --- Qwen 토큰 카운트 확인 ---
        max_len = int(max_sequence_length)
        if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
            raw_ids = pipe.tokenizer(prompt, truncation=False, return_tensors="pt")[
                "input_ids"
            ][0]
            raw_token_count = len(raw_ids)
            clipped = max(0, raw_token_count - max_len)
            if clipped > 0:
                print(
                    f"✗ Qwen 토큰 수: {raw_token_count} / {max_len} → {clipped}개 잘림!"
                )
                truncated_text = pipe.tokenizer.decode(
                    raw_ids[max_len:], skip_special_tokens=True
                )
                print(f"✗ [잘린 텍스트] {truncated_text}")
            else:
                print(f"✓ Qwen 토큰 수: {raw_token_count} / {max_len} (잘림 없음)")

        # Note: Klein uses Qwen3-8B encoder. true_cfg_scale not supported (is_distilled=True).
        print("=" * 60)

        # --- 이미지 준비 (없으면 텍스트-투-이미지로 동작) ---
        if base_image is not None:
            input_images = [base_image.resize((out_w, out_h), Image.LANCZOS).convert("RGB")]
            print("이미지 합성 모드: 기본 이미지 1개 전달")
        else:
            input_images = None
            print("텍스트-투-이미지 모드: 이미지 입력 없음")

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

        pipe_kwargs = {
            "prompt": prompt,
            "image": input_images,  # None이면 텍스트-투-이미지로 동작
            "width": out_w,
            "height": out_h,
            "guidance_scale": 1.0,  # distilled 모델: CFG 미사용, 경고 방지용
            "num_inference_steps": steps,
            "num_images_per_prompt": int(num_images_per_prompt),
            "max_sequence_length": max_len,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        with torch.inference_mode():
            images = pipe(**pipe_kwargs).images

        progress(0.95, desc="이미지 저장 중...")

        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        base_filename = (
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{out_w}x{out_h}"
            f"_step{steps}_seed{int(seed)}_n{int(num_images_per_prompt)}"
        )

        saved_files = []
        for i, image in enumerate(images):
            suffix = f"_img{i + 1}" if len(images) > 1 else ""
            filename = f"{base_filename}{suffix}.{ext}"
            if image_format == "JPEG":
                image.save(filename, format="JPEG", quality=100, subsampling=0)
            else:
                image.save(filename)
            saved_files.append(filename)
            print(f"이미지가 저장되었습니다 : {filename}")

        saved_info = (
            f"저장됨: {saved_files[0]}"
            if len(saved_files) == 1
            else f"{len(saved_files)}장 저장됨: {saved_files[0]} 외"
        )
        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")

        progress(1.0, desc="완료!")
        return (
            make_image_grid(images),
            images,
            saved_files,
            f"✓ 완료! ({elapsed:.1f}초) | {saved_info}",
        )

    except Exception as e:
        return None, [], [], f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()

    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    # Load default base image if path is set
    if DEFAULT_IMAGE_PATH and os.path.isfile(DEFAULT_IMAGE_PATH):
        _default_img = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")
        _init_w, _init_h = get_image_dimensions(_default_img)
        _default_info = f"기본 이미지: {_default_img.width} × {_default_img.height} px  →  출력 크기: {_init_w} × {_init_h} px (16 배수로 반올림)"
    else:
        _default_img = None
        _init_w, _init_h = 768, 1536
        _default_info = "이미지를 업로드하면 원본 크기가 표시됩니다."

    with gr.Blocks(title="Flux.2 Klein 9B Text-to-Image") as interface:
        gr.Markdown("# Flux.2 Klein 9B Text-to-Image")
        gr.Markdown(
            f"텍스트 프롬프트로 이미지를 생성합니다. 기본 이미지를 업로드하면 이미지 합성 모드로 동작합니다."
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

                with gr.Accordion("기본 이미지 (Base Image, 선택사항)", open=_default_img is not None):
                    base_image = gr.Image(
                        label="기본 이미지 (없으면 텍스트-투-이미지로 동작)",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=400,
                        value=_default_img,
                    )
                    image_info = gr.Textbox(
                        label="이미지 정보",
                        value=_default_info,
                        interactive=False,
                    )

                gr.Markdown(
                    "### 프롬프트 구성\n"
                    "- **Positive**: 최종 프롬프트 뒤에 추가됩니다."
                )
                with gr.Accordion(
                    "프롬프트 섹션 (Subject · 포즈 · 의상 · 배경 · 조명 · 카메라)",
                    open=True,
                ):
                    prompt_subject = gr.Textbox(
                        label="1. 주제/대상 (Subject)",
                        value=SUBJECT,
                        lines=2,
                        placeholder="예: 1girl, young woman, a cat",
                        info="이미지의 주된 주제나 대상을 설명합니다.",
                    )
                    prompt_pose_foot = gr.Textbox(
                        label="2. 포즈 - 발 (Foot)",
                        value=FOOT,
                        lines=1,
                        placeholder="예: feet slightly apart, toes pointed forward",
                        info="발의 위치를 설명합니다.",
                    )
                    prompt_pose_leg = gr.Textbox(
                        label="3. 포즈 - 다리 (Leg)",
                        value=LEG,
                        lines=1,
                        placeholder="예: one leg stepping forward, weight on left leg",
                        info="다리 자세를 설명합니다.",
                    )
                    prompt_face = gr.Textbox(
                        label="4. 얼굴/외모 (Face)",
                        value=FACE,
                        lines=2,
                        placeholder="예: fair complexion, blue contact lenses, soft smile",
                        info="얼굴, 피부, 눈, 표정, 머리카락 등을 설명합니다.",
                    )
                    prompt_pose_body = gr.Textbox(
                        label="5. 포즈 - 몸통 (Body)",
                        value=BODY,
                        lines=2,
                        placeholder="예: body angled slightly, leaning forward",
                        info="몸통 자세와 전체 실루엣을 설명합니다.",
                    )
                    prompt_pose_arm = gr.Textbox(
                        label="6. 포즈 - 팔 (Arm)",
                        value=ARM,
                        lines=1,
                        placeholder="예: arms resting across torso",
                        info="팔의 위치와 자세를 설명합니다.",
                    )
                    prompt_pose_hand = gr.Textbox(
                        label="7. 포즈 - 손 (Hand)",
                        value=HAND,
                        lines=1,
                        placeholder="예: one hand gripping the other arm",
                        info="손의 위치와 동작을 설명합니다.",
                    )
                    prompt_footwear = gr.Textbox(
                        label="8. 신발 (Footwear)",
                        value=FOOTWEAR,
                        lines=1,
                        placeholder="예: black stiletto heels, white sneakers",
                        info="신발, 부츠, 샌들 등을 설명합니다.",
                    )
                    prompt_legwear = gr.Textbox(
                        label="9. 레그웨어 (Legwear)",
                        value=LEGWEAR,
                        lines=1,
                        placeholder="예: thigh-high black stockings, sheer tights",
                        info="스타킹, 양말, 레깅스 등을 설명합니다.",
                    )
                    prompt_bottom = gr.Textbox(
                        label="10. 하의 (Bottom)",
                        value=BOTTOM,
                        lines=2,
                        placeholder="예: tiny black panty, mini skirt",
                        info="하의, 속옷 하의 등을 설명합니다.",
                    )
                    prompt_top = gr.Textbox(
                        label="11. 상의 (Top)",
                        value=TOP,
                        lines=2,
                        placeholder="예: sheer black button-up shirt, tiny black bra",
                        info="상의, 속옷 상의 등을 설명합니다.",
                    )
                    prompt_headwear = gr.Textbox(
                        label="12. 머리 장식 (Headwear)",
                        value=HEADWEAR,
                        lines=1,
                        placeholder="예: black beret, floral hairpin",
                        info="모자, 헤어핀, 머리띠 등 머리 장식을 설명합니다.",
                    )
                    prompt_armwear = gr.Textbox(
                        label="13. 팔 장식 (Armwear)",
                        value=ARMWEAR,
                        lines=1,
                        placeholder="예: black lace gloves, silver bracelet",
                        info="장갑, 팔찌, 소매 장식 등을 설명합니다.",
                    )
                    prompt_pose_head = gr.Textbox(
                        label="14. 포즈 - 머리 (Head)",
                        value=HEAD,
                        lines=1,
                        placeholder="예: head tilted slightly, gazing off-camera",
                        info="머리와 시선 방향을 설명합니다.",
                    )
                    prompt_setting = gr.Textbox(
                        label="15. 배경/장소 (Setting)",
                        value=SETTING,
                        lines=2,
                        placeholder="예: rooftop terrace at twilight, city skyline",
                        info="배경, 장소, 환경, 계절 등을 설명합니다.",
                    )
                    prompt_lighting = gr.Textbox(
                        label="16. 조명 (Lighting)",
                        value=LIGHTING,
                        lines=2,
                        placeholder="예: golden hour, city glow, cinematic rim light",
                        info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                    )
                    prompt_camera = gr.Textbox(
                        label="17. 카메라 설정 (Camera)",
                        value=CAMERA,
                        lines=2,
                        placeholder="예: Sony A7R V, 85mm f/1.8, ISO 400",
                        info="카메라 기종, 렌즈, ISO, 셔터 스피드 등을 설명합니다.",
                    )
                    positive_prompt_box = gr.Textbox(
                        label="18. 포지티브 프롬프트 (Positive)",
                        value=POSITIVE,
                        lines=2,
                        placeholder="예: masterpiece, best quality, highly detailed",
                        info="최종 프롬프트 뒤에 추가로 덧붙일 키워드를 입력합니다.",
                    )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            SUBJECT,
                            FACE,
                            HEAD,
                            BODY,
                            ARM,
                            HAND,
                            LEG,
                            FOOT,
                            HEADWEAR,
                            TOP,
                            BOTTOM,
                            LEGWEAR,
                            FOOTWEAR,
                            ARMWEAR,
                            SETTING,
                            LIGHTING,
                            CAMERA,
                        ),
                        lines=4,
                        interactive=False,
                        info="위 섹션들이 자동으로 합쳐진 최종 프롬프트입니다.",
                    )
                prompt_sections = [
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
                        minimum=256,
                        maximum=2048,
                        step=16,
                        value=_init_w,
                        info="출력 이미지 너비 (픽셀).",
                    )
                    height = gr.Slider(
                        label="출력 높이",
                        minimum=256,
                        maximum=2048,
                        step=16,
                        value=_init_h,
                        info="출력 이미지 높이 (픽셀).",
                    )

                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=4,
                        info="Klein distilled 권장: 4~8 스텝.",
                    )

                with gr.Row():
                    num_images_per_prompt = gr.Slider(
                        label="생성 이미지 수",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                        info="한 번에 생성할 이미지 수. 많을수록 VRAM 사용 증가.",
                    )
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )

                with gr.Row():
                    max_sequence_length = gr.Slider(
                        label="최대 시퀀스 길이",
                        minimum=64,
                        maximum=512,
                        step=16,
                        value=MAX_SEQUENCE_LENGTH,
                        info="텍스트 인코더 최대 길이. 긴 프롬프트는 높은 값 필요.",
                    )
                    image_format = gr.Radio(
                        label="이미지 포맷",
                        choices=["JPEG", "PNG"],
                        value="JPEG",
                        info="JPEG: quality 100, PNG: 무손실.",
                    )

                gr.Markdown("### 이미지 생성")
                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")
                output_grid = gr.Image(label="생성된 이미지 (전체 보기)", height=800)
                with gr.Accordion("개별 이미지 다운로드", open=False):
                    output_gallery = gr.Gallery(
                        label="개별 이미지",
                        columns=[1, 1, 2, 2],
                        rows=[1, 1, 1, 2],
                        object_fit="contain",
                        allow_preview=True,
                    )
                    output_files = gr.Files(label="파일 다운로드")
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
                base_image,
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
                num_inference_steps,
                num_images_per_prompt,
                seed,
                positive_prompt_box,
                max_sequence_length,
                image_format,
            ],
            outputs=[output_grid, output_gallery, output_files, output_message],
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
