import re
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

# 섹션 기본값 (Flux Example51-R01과 동일 순서로 한 문자열로 결합)
SUBJECT = "A full body photography of a beautiful young skinny Korean woman with soft idol aesthetics standing on a casual spring outing in Seoul."

FOOT = "Both feet pressed firmly together with ankles touching, inner sides of both feet in contact, toes pointing straight forward, white sneakers clearly visible side by side, feet entirely in frame and not cropped."

LEG = "Both legs straight and pressed firmly together, knees touching, thighs together, no gap between legs."

FACE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious in the eyes, with a faint soft smile and lips gently closed, looking directly at the camera. She has long wavy voluminous jet-black hair with beautiful soft waves and curls, hair entirely on the back."

BODY = "Standing perfectly still and upright, both feet and legs together, body facing completely straight toward the camera, chest and torso fully frontal, posture tall and elegant, shoulders back."

ARM = "Both arms hanging naturally and relaxed at her sides."

HAND = "Both hands hanging gracefully at her sides with fingers lightly extended."

FOOTWEAR = "Clean white canvas sneakers, simple and casual."

LEGWEAR = "Bare legs, smooth and fair skin visible through the skirt slit, legs themselves pressed together."

BOTTOM = ""

TOP = "Dark navy chiffon long one-piece dress with thin spaghetti straps, simple neckline, bare shoulders and arms, mostly opaque with only a slight translucency, densely scattered tiny cherry blossom print in soft pink and white, fitted waist, flowing A-line skirt with a side slit slightly opened by a very calm light breeze, skirt hem and fabric lifted only a little on one side, subtle flutter, revealing the bare leg while both legs remain firmly pressed together."

HEADWEAR = ""

ARMWEAR = ""

HEAD = "Head tilted very slightly to one side, gentle and relaxed posture."

SETTING = "Bright spring street in Seoul, cherry blossom trees lining the sidewalk with pink petals falling gently, warm sunny day, clean pavement."

LIGHTING = "Bright even spring daylight, soft frontal natural light, face clearly and brightly lit, no harsh shadows."

CAMERA = "Full body shot, entire body from head to feet fully in frame, feet and sneakers not cropped, waist level angle, subject facing camera, sharp focus, soft bokeh background."

POSITIVE = "8k, high quality, realistic, detailed, sharp focus, perfect anatomy, ten fingers."

NEGATIVE = "Blurry, low quality, deformed, bad anatomy, extra limbs, ugly, watermark, text, signature, extra fingers, one leg forward, staggered legs, walking pose, weight shift, legs apart, stepping, feet apart, spread legs, gap between feet, gap between legs, wide stance, smile, smiling, grin, grinning, laugh, laughing, cheerful expression, happy mouth, teeth showing in a smile."


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    subject,
    foot,
    leg,
    face,
    body,
    arm,
    hand,
    footwear,
    legwear,
    bottom,
    top,
    headwear,
    armwear,
    head,
    setting,
    lighting,
    camera,
    positive,
):
    """섹션을 순서대로 하나의 프롬프트 문자열로 합칩니다."""
    sections = [
        subject,
        foot,
        leg,
        face,
        body,
        arm,
        hand,
        footwear,
        legwear,
        bottom,
        top,
        headwear,
        armwear,
        head,
        setting,
        lighting,
        camera,
        positive,
    ]
    combined = " ".join(normalize_spacing(s) for s in sections if s and s.strip())
    return combined


def z_image_token_count(pipe, user_content: str) -> int:
    """ZImagePipeline._encode_prompt와 동일한 chat_template 뒤 토큰 수(잘림 전)."""
    messages = [{"role": "user", "content": user_content}]
    tok = pipe.tokenizer
    try:
        formatted = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        formatted = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return len(tok.encode(formatted))


def describe_prompt_truncation(
    pipe,
    prompt_text: str,
    negative_text: str | None,
    max_sequence_length: int,
) -> str:
    """프롬프트/네거티브 프롬프트 각각의 토큰 수와 max_sequence_length 대비 잘림 여부를 표시."""
    if pipe is None:
        return "모델 미로드: 토큰 수는 모델 로드 후 표시됩니다."
    max_seq = max(1, int(max_sequence_length))
    pos = prompt_text or ""
    neg = negative_text or ""

    def line(label: str, n: int) -> str:
        if n <= max_seq:
            return f"{label}: {n} / {max_seq} 토큰 — 잘림 없음"
        over = n - max_seq
        return (
            f"{label}: {n} / {max_seq} 토큰 — 잘림 예상 "
            f"(약 {over}토큰 초과, truncation=True로 끝부분 제거)"
        )

    try:
        n_pos = z_image_token_count(pipe, pos)
        n_neg = z_image_token_count(pipe, neg)
    except Exception as e:
        return f"토큰 계산 오류: {e}"

    return line("프롬프트", n_pos) + "\n" + line("네거티브 프롬프트", n_neg)


def compact_truncation_hint(
    pipe,
    prompt_text: str,
    negative_text: str | None,
    max_sequence_length: int,
) -> str:
    """상태 메시지 한 줄용."""
    if pipe is None:
        return ""
    max_seq = max(1, int(max_sequence_length))
    try:
        n_pos = z_image_token_count(pipe, (prompt_text or "").strip())
        n_neg = z_image_token_count(pipe, (negative_text or "").strip())
    except Exception:
        return ""
    parts = [
        f"프롬프트 {n_pos}/{max_seq}",
        f"네거티브 프롬프트 {n_neg}/{max_seq}",
    ]
    warn = []
    if n_pos > max_seq:
        warn.append(f"프롬프트 -{n_pos - max_seq}")
    if n_neg > max_seq:
        warn.append(f"네거티브 프롬프트 -{n_neg - max_seq}")
    if warn:
        return "토큰 잘림: " + ", ".join(warn)
    return "토큰: " + ", ".join(parts) + " OK"


def refresh_prompt_token_ui(combined: str, neg: str, max_seq: float):
    """Gradio: 프롬프트/네거티브 프롬프트/길이 변경 시 토큰·잘림 표시."""
    return describe_prompt_truncation(pipe, combined, neg or "", int(max_seq))


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


def get_device_tag_for_filename(device: str) -> str:
    """Return a human-readable device tag for output filename."""
    if device == "cuda" and torch.cuda.is_available():
        try:
            current_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_idx).strip()
            props = torch.cuda.get_device_properties(current_idx)
            vram_gb = props.total_memory / (1024**3)
            vram_tag = f"{vram_gb:.1f}".rstrip("0").rstrip(".") + "GB"
            # Keep the original GPU name as much as possible, but avoid path-breaking chars.
            clean_name = re.sub(r"\bgeforce\b", "", gpu_name, flags=re.IGNORECASE)
            clean_name = (
                clean_name.replace("/", "-")
                .replace(":", "-")
                .replace(" ", "_")
                .strip("_- ")
            )
            return f"{clean_name}_{vram_tag}"
        except Exception:
            return "CUDA"
    if device == "mps":
        return "MPS"
    return str(device).upper()


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
    max_sequence_length,
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
        msl = int(max_sequence_length)
        trunc_report = describe_prompt_truncation(
            pipe, prompt, negative_prompt or "", msl
        )
        trunc_short = compact_truncation_hint(
            pipe, prompt, negative_prompt or "", msl
        )
        print("[프롬프트 토큰 / 잘림]")
        print(trunc_report)
        if trunc_short:
            print(f"[요약] {trunc_short}")

        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="이미지 생성 준비 중...")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        progress(0.05, desc="추론 시작...")

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
                f"[{elapsed:.1f}s 경과, ETA {eta:.1f}s, "
                f"이번 스텝 {step_time:.2f}s, 평균 {avg_speed:.2f}s/스텝]"
            )
            sys.stdout.flush()
            if current == steps:
                print()

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
            max_sequence_length=msl,
        ).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        device_tag = get_device_tag_for_filename(DEVICE)
        ext = "jpg" if image_format == "JPEG" else "png"
        filename = (
            f"{script_name}_{timestamp}_{device_tag}_{width}x{height}"
            f"_gs{guidance_scale}_step{steps}_cfgnorm{cfg_normalization}_seed{int(seed)}.{ext}"
        )

        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")
        print(f"이미지가 저장되었습니다 : {filename}")
        if image_format == "JPEG":
            image.save(filename, format="JPEG", quality=100, subsampling=0)
        else:
            image.save(filename)

        progress(1.0, desc="완료!")
        status_tail = f" | {trunc_short}" if trunc_short else ""
        return (
            image,
            f"✓ 완료! ({elapsed:.1f}초) | 저장됨: {filename}{status_tail}",
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

                gr.Markdown(
                    "### 프롬프트 구성\n"
                    "섹션을 **Subject → … → Camera → Positive** 순으로 합친 문자열을 생성에 사용합니다."
                )
                with gr.Accordion(
                    "프롬프트 섹션 (1–18: Subject … Camera · 프롬프트)",
                    open=True,
                ):
                    prompt_subject = gr.Textbox(
                        label="1. 주제/대상 (Subject)",
                        value=SUBJECT,
                        lines=2,
                        placeholder="예: 1girl, young woman, a cat",
                        info="프롬프트 맨 앞에 올 주제 문장입니다.",
                    )
                    prompt_foot = gr.Textbox(
                        label="2. 포즈 - 발 (Foot)",
                        value=FOOT,
                        lines=1,
                        placeholder="예: feet slightly apart, toes pointed forward",
                        info="발의 위치를 설명합니다.",
                    )
                    prompt_leg = gr.Textbox(
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
                        placeholder="예: fair complexion, blue contact lenses, neutral mouth no smile",
                        info="얼굴, 피부, 눈, 표정, 머리카락 등을 설명합니다.",
                    )
                    prompt_body = gr.Textbox(
                        label="5. 포즈 - 몸통 (Body)",
                        value=BODY,
                        lines=2,
                        placeholder="예: body angled slightly, leaning forward",
                        info="몸통 자세와 전체 실루엣을 설명합니다.",
                    )
                    prompt_arm = gr.Textbox(
                        label="6. 포즈 - 팔 (Arm)",
                        value=ARM,
                        lines=1,
                        placeholder="예: arms resting across torso",
                        info="팔의 위치와 자세를 설명합니다.",
                    )
                    prompt_hand = gr.Textbox(
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
                        info="상의, 원피스·드레스 등을 설명합니다.",
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
                    prompt_head = gr.Textbox(
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
                    positive_box = gr.Textbox(
                        label="18. 프롬프트 (Positive)",
                        value=POSITIVE,
                        lines=2,
                        placeholder="예: masterpiece, best quality, highly detailed",
                        info="품질·디테일 키워드. 결합 문자열 맨 끝에 붙습니다.",
                    )
                with gr.Accordion("최종 프롬프트 (전체 섹션 결합)", open=False):
                    combined_prompt = gr.Textbox(
                        label="Z-Image 프롬프트",
                        value=combine_prompt_sections(
                            SUBJECT,
                            FOOT,
                            LEG,
                            FACE,
                            BODY,
                            ARM,
                            HAND,
                            FOOTWEAR,
                            LEGWEAR,
                            BOTTOM,
                            TOP,
                            HEADWEAR,
                            ARMWEAR,
                            HEAD,
                            SETTING,
                            LIGHTING,
                            CAMERA,
                            POSITIVE,
                        ),
                        lines=5,
                        interactive=True,
                        info="섹션 편집 시 자동 갱신됩니다. 필요하면 여기서 직접 수정할 수 있습니다.",
                    )
                with gr.Accordion("네거티브 프롬프트 (Negative)", open=False):
                    negative_box = gr.Textbox(
                        label="네거티브 프롬프트 (Negative Prompt)",
                        value=NEGATIVE,
                        lines=2,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="Z-Image 파이프라인의 negative_prompt로 전달됩니다.",
                    )
                prompt_sections = [
                    prompt_subject,
                    prompt_foot,
                    prompt_leg,
                    prompt_face,
                    prompt_body,
                    prompt_arm,
                    prompt_hand,
                    prompt_footwear,
                    prompt_legwear,
                    prompt_bottom,
                    prompt_top,
                    prompt_headwear,
                    prompt_armwear,
                    prompt_head,
                    prompt_setting,
                    prompt_lighting,
                    prompt_camera,
                    positive_box,
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
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=768,
                        info="64의 배수. Z-Image 권장 해상도 범위(총 픽셀): 512×512 ~ 2048×2048.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1536,
                        info="64의 배수. 세로 풀샷 등은 768×1536 전후가 무난합니다.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=4.0,
                        info="공식 권장 3.0–5.0. 기본 4.0 (Tongyi-MAI/Z-Image README).",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25, # 50 for original code.
                        info="공식 권장 28–50. 기본 50 (README 예제와 동일, 품질 우선).",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    cfg_normalization = gr.Checkbox(
                        label="CFG Normalization",
                        value=False,
                        info="공식 예제는 False. True는 취향/실사 쪽에 맞출 때 시도.",
                    )

                max_sequence_length = gr.Slider(
                    label="텍스트 최대 시퀀스 길이 (max_sequence_length)",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024,
                    info="기본 1024. 토큰이 이보다 길면 잘립니다. (diffusers 파이프라인 기본값은 512)",
                )
                prompt_token_status = gr.Textbox(
                    label="프롬프트 토큰 / 잘림",
                    value="",
                    lines=4,
                    interactive=False,
                    info="파이프라인과 동일한 chat_template 후 tokenizer.encode 길이로 판단합니다.",
                )

                gr.Markdown("---")
                gr.Markdown("### 이미지 생성")
                image_format = gr.Radio(
                    label="이미지 포맷",
                    choices=["JPEG", "PNG"],
                    value="JPEG",
                    info="JPEG: 용량 작음 (권장), PNG: 무손실",
                )
                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")
                output_image = gr.Image(label="생성된 이미지", height=700)
                output_message = gr.Textbox(label="상태", interactive=False)

        # Load model when button is clicked
        load_model_btn.click(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        ).then(
            fn=refresh_prompt_token_ui,
            inputs=[combined_prompt, negative_box, max_sequence_length],
            outputs=[prompt_token_status],
        )

        # Auto-load model when device is changed
        device_selector.change(
            fn=load_model,
            inputs=[device_selector],
            outputs=[device_status],
        ).then(
            fn=refresh_prompt_token_ui,
            inputs=[combined_prompt, negative_box, max_sequence_length],
            outputs=[prompt_token_status],
        )

        for trig in (combined_prompt, negative_box, max_sequence_length):
            trig.change(
                fn=refresh_prompt_token_ui,
                inputs=[combined_prompt, negative_box, max_sequence_length],
                outputs=[prompt_token_status],
            )

        interface.load(
            fn=refresh_prompt_token_ui,
            inputs=[combined_prompt, negative_box, max_sequence_length],
            outputs=[prompt_token_status],
        )

        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_image,
            inputs=[
                combined_prompt,
                negative_box,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                cfg_normalization,
                max_sequence_length,
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
