import re
import torch
import platform
from diffusers import FluxKontextPipeline

# Open-weights image editing model (non-commercial license; HF gated).
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
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
SUBJECT = ""
FOOT = (
    "Both feet close together in a narrow stance, parallel feet, "
    "heels and toes gathered without a wide spread"
)
LEG = (
    "One knee slightly bent, body weight supported on one leg, "
    "the other leg relaxed with a soft bend at the knee"
)
FACE = ""
BODY = ""
ARM = ""
HAND = ""
FOOTWEAR = "Wearing a black high heel sandals."
LEGWEAR = ""
BOTTOM = ""
TOP = ""
HEADWEAR = ""
ARMWEAR = ""
HEAD = ""
SETTING = (
    "Five-star luxury hotel outdoor swimming pool, upscale resort pool deck, "
    "refined lounge chairs and tiles, sparkling clear water, "
    "standing pose at the poolside on the deck near the water's edge"
)
LIGHTING = ""
CAMERA = "Waist level angle shot, long legs."
POSITIVE = "Masterpiece, best quality, highly detailed, sharp focus, natural lighting, photorealistic, faithful to the reference image, anatomically correct hands and feet, five fingers per hand, five toes per foot, clearly separated fingers and toes."
NEGATIVE = "Low quality, worst quality, blurry, out of focus, jpeg artifacts, distorted, deformed, bad anatomy, watermark, text, logo, extra fingers, missing fingers, fused fingers, malformed hands, wrong finger count, extra toes, missing toes, fused toes, malformed feet, wrong toe count, mitten hands, claw hands, deformed nails"

# diffusers FLUX / FluxKontextPipeline: width·height must be divisible by 8 (pipeline check_inputs).
KONTEXT_SPATIAL_MULTIPLE = 8
KONTEXT_SIZE_MIN = 256
KONTEXT_SIZE_MAX = 2048


def round_to_kontext_spatial(
    value: float | int,
    minimum: int = KONTEXT_SIZE_MIN,
    maximum: int = KONTEXT_SIZE_MAX,
    multiple: int = KONTEXT_SPATIAL_MULTIPLE,
) -> int:
    """Clamp to [minimum, maximum] and round to the nearest supported pixel multiple."""
    v = int(round(float(value)))
    m = int(multiple)
    v = max(int(minimum), min(int(maximum), v))
    if m <= 1:
        return v
    r = int(round(v / m) * m)
    r = max(int(minimum), min(int(maximum), r))
    if r % m != 0:
        r = (r // m) * m
    if r < int(minimum):
        r = (int(minimum) + m - 1) // m * m
    if r > int(maximum):
        r = int(maximum) // m * m
    return max(m, r)


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


def format_clip_preview(subject: str, positive: str) -> str:
    """CLIP 미리보기: Subject 뒤에 Positive."""
    s = normalize_spacing(subject) if subject and subject.strip() else ""
    p = normalize_spacing(positive) if positive and positive.strip() else ""
    if s and p:
        return s + " " + p
    return s or p


def combine_t5_prompt_sections(
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
    """Combine separate prompt sections into the T5 (prompt_2) string."""
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


def combine_prompt_sections_dual(
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
    """Build CLIP / T5 문자열 (FluxKontextPipeline의 prompt / prompt_2에 그대로 전달)."""
    clip_prompt = format_clip_preview(subject, positive)
    t5_prompt = combine_t5_prompt_sections(
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
    )
    return clip_prompt, t5_prompt


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
    """Load FLUX.1-Kontext-dev (FluxKontextPipeline) with memory optimizations."""
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
    print(f"FLUX.1-Kontext-dev 이미지 편집 파이프라인 로드: {MODEL_ID}")
    pipe = FluxKontextPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

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
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)
        elif hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
        print("메모리 최적화 적용: attention slicing, VAE slicing, VAE tiling (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    input_image,
    prompt_clip,
    prompt_t5,
    negative,
    negative_prompt_2,
    width,
    height,
    guidance_scale,
    true_cfg_scale,
    num_inference_steps,
    num_images_per_prompt,
    seed,
    max_sequence_length,
    max_area,
    kontext_auto_resize,
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

    if input_image is None:
        return None, [], [], "오류: 편집할 입력 이미지를 업로드해 주세요."

    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        pc = (prompt_clip or "").strip()
        pt = (prompt_t5 or "").strip()
        if not pc and not pt:
            return (
                None,
                [],
                [],
                "오류: CLIP 프롬프트 또는 T5 프롬프트 중 하나 이상을 입력해 주세요.",
            )

        clip_prompt_text = pc if pc else pt
        t5_prompt_text = pt if pt else pc

        progress(0.0, desc="프롬프트 분석 · 추론 준비...")
        print("FLUX.1-Kontext-dev 편집 호출")
        print("=" * 60)
        print("[prompt → CLIP / pooled]")
        print(clip_prompt_text)
        print("-" * 60)
        print("[prompt_2 → T5]")
        print(t5_prompt_text)
        print("=" * 60)

        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        max_len = int(max_sequence_length)
        raw_ids = pipe.tokenizer_2(
            t5_prompt_text, truncation=False, return_tensors="pt"
        )["input_ids"][0]
        raw_token_count = len(raw_ids)
        clipped = max(0, raw_token_count - max_len)
        if clipped > 0:
            print(f"✗ T5 토큰 수: {raw_token_count} / {max_len} → {clipped}개 잘림!")
            truncated_text = pipe.tokenizer_2.decode(
                raw_ids[max_len:], skip_special_tokens=True
            )
            print("-" * 60)
            print("✗ [잘린 T5 텍스트]")
            print(truncated_text)
            print("-" * 60)
        else:
            print(f"✓ T5 토큰 수: {raw_token_count} / {max_len} (잘림 없음)")

        w_px = round_to_kontext_spatial(width)
        h_px = round_to_kontext_spatial(height)
        if w_px != int(width) or h_px != int(height):
            print(
                f"이미지 크기를 Kontext 지원 해상도({KONTEXT_SPATIAL_MULTIPLE}px 배수)로 맞춤: "
                f"{int(width)}x{int(height)} → {w_px}x{h_px}"
            )

        resized_input = input_image.resize((w_px, h_px), Image.LANCZOS)

        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.90
            progress(
                progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)"
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
            "image": resized_input,
            "prompt": clip_prompt_text,
            "prompt_2": t5_prompt_text,
            "width": w_px,
            "height": h_px,
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": steps,
            "num_images_per_prompt": int(num_images_per_prompt),
            "generator": generator,
            "callback_on_step_end": step_callback,
            "max_sequence_length": max_len,
            "max_area": int(max_area),
            "_auto_resize": bool(kontext_auto_resize),
        }
        neg = (negative or "").strip()
        neg2 = (negative_prompt_2 or "").strip()
        if float(true_cfg_scale) > 1.0 and neg:
            pipe_kwargs["negative_prompt"] = neg
            pipe_kwargs["negative_prompt_2"] = neg2 if neg2 else neg
            pipe_kwargs["true_cfg_scale"] = float(true_cfg_scale)

        progress(0.05, desc="추론 시작...")

        with torch.inference_mode():
            images = pipe(**pipe_kwargs).images

        progress(0.95, desc="이미지 저장 중...")

        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        if DEVICE == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
            gpu_label = (
                gpu_name.replace(" ", "").replace("NVIDIA", "").replace("GeForce", "")
                + f"-{gpu_mem}GB"
            )
            device_label = gpu_label
        else:
            device_label = DEVICE.upper()
        base_filename = (
            f"{script_name}_{timestamp}_{device_label}_{w_px}x{h_px}"
            f"_gs{guidance_scale}_step{steps}_seed{int(seed)}"
            f"_cfg{true_cfg_scale}_n{int(num_images_per_prompt)}_msl{int(max_sequence_length)}"
            f"_ma{int(max_area)}_ar{1 if kontext_auto_resize else 0}"
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

        token_info = (
            f"T5 토큰: {raw_token_count}/{max_len} → {clipped}개 잘림!"
            if clipped > 0
            else f"T5 토큰: {raw_token_count}/{max_len}"
        )
        saved_info = (
            f"저장됨: {saved_files[0]}"
            if len(saved_files) == 1
            else f"{len(saved_files)}장 저장됨: {saved_files[0]} 외"
        )
        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초 | {token_info}")

        try:
            w_in = int(round(float(width)))
            h_in = int(round(float(height)))
        except (TypeError, ValueError):
            w_in, h_in = w_px, h_px
        snap_msg = ""
        if w_in != w_px or h_in != h_px:
            snap_msg = (
                f" | 해상도 {w_in}×{h_in} → {w_px}×{h_px} "
                f"({KONTEXT_SPATIAL_MULTIPLE}px 배수로 보정)"
            )

        progress(1.0, desc="완료!")
        return (
            make_image_grid(images),
            images,
            saved_files,
            f"✓ 완료! ({elapsed:.1f}초) | {token_info} | {saved_info}{snap_msg}",
        )
    except Exception as e:
        return None, [], [], f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()
    load_model()

    with gr.Blocks(title="Flux.1-Kontext-dev Image Edit") as interface:
        gr.Markdown("# Flux.1-Kontext-dev Image Edit")
        gr.Markdown(
            f"`{MODEL_ID}` — 텍스트 지시로 **입력 이미지를 편집**합니다. "
            "diffusers는 `prompt`→CLIP(pooled), `prompt_2`→T5 순으로 인코딩합니다. "
            "공식 예시와 같이 **Guidance 2.5~3.5**, **약 28 스텝**부터 맞춰 보세요. "
            "고급 옵션은 오른쪽 **FluxKontext 전용** 아코디언(`max_area`, `_auto_resize`, `negative_prompt_2`)을 참고하세요. "
            f"(Device: **{DEVICE.upper()}**)"
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
                input_image = gr.Image(
                    label="편집할 이미지",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=300,
                )

                gr.Markdown(
                    "### 프롬프트 구성\n"
                    "- **CLIP 미리보기** → 파이프라인 `prompt` (짧은 태그·요약에 적합, ~77 토큰).\n"
                    "- **T5 미리보기** → 파이프라인 `prompt_2` (편집 지시·세부 묘사, 최대 시퀀스 길이까지).\n"
                    "- **Negative**: `True CFG > 1.0` 이고 네거티브가 비어 있지 않을 때 `negative_prompt` / `negative_prompt_2`로 전달."
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
                        info="T5 프롬프트 맨 앞. CLIP은 Subject → Positive 순.",
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
                        placeholder="예: fair complexion, blue contact lenses, soft smile",
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
                        label="18. 포지티브 프롬프트 (Positive)",
                        value=POSITIVE,
                        lines=2,
                        placeholder="예: masterpiece, best quality, highly detailed",
                        info="CLIP과 T5 모두에 포함됩니다. 편집 지시는 Subject/T5와 함께 여기에 두면 반응이 좋아질 수 있습니다.",
                    )
                    negative_box = gr.Textbox(
                        label="19. 네거티브 프롬프트 (negative_prompt)",
                        value=NEGATIVE,
                        lines=2,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="True CFG > 1.0일 때 CLIP 쪽 네거티브. T5 전용 문구는 오른쪽 Kontext 섹션의 negative_prompt_2.",
                    )
                with gr.Accordion("CLIP 프롬프트 (Subject → Positive)", open=False):
                    prompt_clip = gr.Textbox(
                        label="CLIP 프롬프트",
                        value=format_clip_preview(SUBJECT, POSITIVE),
                        lines=3,
                        interactive=True,
                        info="Subject + Positive 순으로 자동 결합됩니다.",
                    )
                with gr.Accordion("T5 프롬프트 (전체 섹션 결합)", open=False):
                    prompt_t5 = gr.Textbox(
                        label="T5 프롬프트",
                        value=combine_t5_prompt_sections(
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
                        info="Subject → 포즈·의상·머리 → Setting → Lighting → Camera → Positive 순으로 자동 결합됩니다.",
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
                        fn=combine_prompt_sections_dual,
                        inputs=prompt_sections,
                        outputs=[prompt_clip, prompt_t5],
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=KONTEXT_SIZE_MIN,
                        maximum=KONTEXT_SIZE_MAX,
                        step=KONTEXT_SPATIAL_MULTIPLE,
                        value=768,
                        info=(
                            f"픽셀. FLUX Kontext(diffusers)는 가로·세로가 "
                            f"{KONTEXT_SPATIAL_MULTIPLE}의 배수여야 합니다."
                        ),
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=KONTEXT_SIZE_MIN,
                        maximum=KONTEXT_SIZE_MAX,
                        step=KONTEXT_SPATIAL_MULTIPLE,
                        value=1536,
                        info=(
                            f"픽셀. FLUX Kontext(diffusers)는 가로·세로가 "
                            f"{KONTEXT_SPATIAL_MULTIPLE}의 배수여야 합니다."
                        ),
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=7.5,
                        info="Kontext 공식 예시: ~2.5~3.5. 높이면 지시 준수↑, 과하면 아티팩트 가능.",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28,
                        info="공식 예시: 28. 품질 우선 시 40~50 (시간 증가).",
                    )
                with gr.Row():
                    true_cfg_scale = gr.Slider(
                        label="True CFG Scale (네거티브 프롬프트 강도)",
                        minimum=1.0,
                        maximum=5.0,
                        step=0.5,
                        value=1.5,
                        info="1.0이면 네거티브 프롬프트 비활성화. 1.5~2.0 권장.",
                    )
                    num_images_per_prompt = gr.Slider(
                        label="생성 이미지 수",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                        info="한 번에 생성할 이미지 수. 많을수록 VRAM 사용 증가.",
                    )
                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    max_sequence_length = gr.Slider(
                        label="최대 시퀀스 길이",
                        minimum=64,
                        maximum=512,
                        step=64,
                        value=512,
                        info="텍스트 인코더 최대 길이. 긴 프롬프트는 높은 값 필요.",
                    )
                with gr.Row():
                    image_format = gr.Radio(
                        label="이미지 포맷",
                        choices=["JPEG", "PNG"],
                        value="JPEG",
                        info="JPEG: quality 100 (4:4:4), PNG: 무손실 압축.",
                    )

                with gr.Accordion(
                    "FluxKontext 전용 (diffusers `FluxKontextPipeline`)", open=False
                ):
                    gr.Markdown(
                        "`max_area`: 출력 해상도(너비·높이 슬라이더 비율 유지)의 **픽셀 면적 상한**. "
                        "`_auto_resize`: 입력 이미지를 **학습에 쓰인 권장 해상도** 중 하나로 맞춤. "
                        "`negative_prompt_2`: **T5 전용** 네거티브(비우면 `negative_prompt`와 동일)."
                    )
                    max_area = gr.Slider(
                        label="max_area (최대 픽셀 수, width×height 상한의 기준)",
                        minimum=262144,
                        maximum=4194304,
                        step=65536,
                        value=1048576,
                        info="기본 1024². 비율은 좌측 너비/높이 슬라이더에 맞춰 조정됩니다.",
                    )
                    kontext_auto_resize = gr.Checkbox(
                        label="_auto_resize (권장 해상도로 입력 이미지 정렬)",
                        value=True,
                        info="켜면 파이프라인이 Kontext 학습 해상도 목록에 맞춰 입력을 리사이즈합니다.",
                    )
                    negative_prompt_2_box = gr.Textbox(
                        label="negative_prompt_2 (T5 전용, 선택)",
                        value="",
                        lines=2,
                        placeholder="비우면 negative_prompt(19번)와 동일하게 전달",
                        info="True CFG > 1.0 이고 네거티브가 있을 때만 의미가 있습니다.",
                    )

                gr.Markdown("---")
                gr.Markdown("### 이미지 생성")
                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")
                output_grid = gr.Image(label="생성된 이미지 (전체 보기)", height=700)
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
                negative_prompt_2_box,
                width,
                height,
                guidance_scale,
                true_cfg_scale,
                num_inference_steps,
                num_images_per_prompt,
                seed,
                max_sequence_length,
                max_area,
                kontext_auto_resize,
                image_format,
            ],
            outputs=[output_grid, output_gallery, output_files, output_message],
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
