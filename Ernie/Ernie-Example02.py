import re
import inspect
import torch
import platform
import logging
from diffusers import ErnieImagePipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# Suppress harmless type-mismatch warnings from diffusers
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Default values for each prompt section
SUBJECT = "A photorealistic full-body portrait of a beautiful slender young Korean woman with soft idol-like features, standing upright on a sunny beach in a very tiny white thong bikini, entire body from head to bare feet fully visible in frame."

FACE = "Fair glowing complexion with a light sun-kissed warmth, striking bright blue contact lenses, long straight jet-black hair falling naturally over her shoulders, lips gently closed in a soft smile, looking directly at the camera."

HEAD = "Head held upright with a very slight tilt to one side, calm and confident gaze."

BODY = "Standing fully upright and facing the camera, subtle S-curve posture with weight shifted onto one straight leg, slim waist and flat stomach clearly visible."

TOP = "Very tiny white triangle bikini top with thin shoulder straps and minimal coverage."

BOTTOM = "Very tiny white thong bikini bottom with minimal coverage."

HEADWEAR = ""

ARMWEAR = "Delicate gold chain necklace resting at the neckline, thin gold bracelet on one wrist."

ARM = "Both arms hanging naturally and relaxed at her sides."

HAND = "Both hands relaxed at her sides, fingers gently extended."

LEGWEAR = ""

LEG = "Both long straight legs pressed firmly together, inner thighs and knees touching, no gap between legs."

FOOT = "Both bare feet flat on the sand, feet together with heels and inner edges touching, all ten toes clearly visible."

FOOTWEAR = ""

SETTING = "Sunny tropical beach, clear turquoise ocean in the background, white sandy ground, bright blue sky with a few soft clouds."

LIGHTING = "Bright warm frontal sunlight evenly illuminating the entire body, face and torso clearly lit with no shadows falling on the front."

CAMERA = "Full-body shot from head to toes, 85mm portrait lens at f/1.8, tack-sharp focus on the subject, softly blurred ocean background."

POSITIVE = "Photorealistic, 8K resolution, perfect anatomy, complete full body in frame, all ten toes visible, beautiful detailed face."

NEGATIVE = "Blurry, low quality, deformed, bad anatomy, extra limbs, missing limbs, watermark, text, signature, cropped feet, cropped body, shoes, open mouth, legs apart, feet apart, floating, disconnected limbs."


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
    """Combine all prompt sections into a single ERNIE prompt string.

    Order: Subject → Face → Head → Body → Top → Bottom → Headwear → Armwear
           → Arm → Hand → Legwear → Leg → Foot → Footwear → Setting → Lighting → Camera → Positive
    (위에서 아래로, 중심에서 주변부 순서로 자연스럽게 읽히도록 배치)
    """
    sections = [
        subject,
        face,
        head,
        body,
        top,
        bottom,
        headwear,
        armwear,
        arm,
        hand,
        legwear,
        leg,
        foot,
        footwear,
        setting,
        lighting,
        camera,
        positive,
    ]
    combined = " ".join(normalize_spacing(s) for s in sections if s and s.strip())
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
    """Load and initialize the ERNIE model with optimizations."""
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
    pretrained_kwargs = {"torch_dtype": DTYPE}
    try:
        pipe = ErnieImagePipeline.from_pretrained(
            "Baidu/ERNIE-Image",
            local_files_only=True,
            **pretrained_kwargs,
        )
        print("캐시에서 모델 로드됨 (다운로드 없음)")
    except Exception:
        print("캐시된 모델 없음 — HuggingFace Hub에서 다운로드 중...")
        pipe = ErnieImagePipeline.from_pretrained(
            "Baidu/ERNIE-Image",
            **pretrained_kwargs,
        )
        print("다운로드 및 캐싱 완료")
    pipe.to(DEVICE)

    if DEVICE == "cuda":
        # enable_model_cpu_offload 만 사용: sequential_cpu_offload는 _execution_device를
        # cpu로 바꿔 PE/text_encoder의 input_ids device 불일치 경고를 유발함.
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: model CPU offload, attention slicing (CUDA)")
    elif DEVICE == "cpu":
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (CPU)")
    elif DEVICE == "mps":
        pipe.enable_attention_slicing()
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)
        elif hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
        print("메모리 최적화 적용: attention slicing (MPS)")

    # ---- 파이프라인에서 권장값 읽기 ----
    sig = inspect.signature(pipe.__call__)

    def _default(name, fallback):
        p = sig.parameters.get(name)
        return p.default if (p and p.default is not inspect.Parameter.empty) else fallback

    rec_width        = _default("width", 1024)
    rec_height       = _default("height", 1024)
    rec_guidance     = _default("guidance_scale", 4.0)
    rec_steps        = _default("num_inference_steps", 50)
    rec_num_images   = _default("num_images_per_prompt", 1)
    rec_use_pe       = _default("use_pe", True)

    vae_sf  = pipe.vae_scale_factor
    tok_max = getattr(pipe.tokenizer, "model_max_length", None) or 2048
    rec_max_seq = min(tok_max, 1024)

    param_info_text = (
        f"- **이미지 크기**: **{vae_sf}의 배수** (VAE scale factor = {vae_sf}). "
        f"권장: {rec_width}×{rec_height} (정방형), 848×1264 (세로), 1264×848 (가로).\n"
        f"- **최대 시퀀스 길이**: 토크나이저 최대 = **{tok_max}** 토큰. "
        f"슬라이더 범위(64~512) 내에서 설정. 초과 토큰은 잘림."
    )

    status = (
        f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE}) | "
        f"VAE scale: {vae_sf} | 토크나이저 최대: {tok_max} 토큰"
    )
    print(status)
    return (
        status,
        gr.update(value=rec_width,   step=vae_sf),
        gr.update(value=rec_height,  step=vae_sf),
        gr.update(value=rec_guidance),
        gr.update(value=rec_steps),
        gr.update(value=rec_num_images),
        gr.update(value=rec_use_pe),
        gr.update(value=rec_max_seq, maximum=tok_max),
        gr.update(value=rec_max_seq, maximum=tok_max),
        param_info_text,
    )


def _encode_prompt(text, max_len, num_images):
    """파이프라인의 encode_prompt를 사용해 임베딩 리스트를 반환.

    pipe.text_encoder를 직접 호출하면 accelerate cpu_offload 훅을 우회해
    device 불일치가 발생한다. 파이프라인 내부 메서드를 통해 호출하면
    훅이 정상 동작하여 device를 자동으로 관리한다.
    tokenizer.model_max_length를 임시로 교체해 max_len을 적용한다.
    """
    orig_max = pipe.tokenizer.model_max_length
    pipe.tokenizer.model_max_length = max_len
    try:
        device = pipe._execution_device
        return pipe.encode_prompt(text, device, num_images)
    finally:
        pipe.tokenizer.model_max_length = orig_max


def generate_image(
    prompt,
    negative,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    num_images_per_prompt,
    seed,
    use_pe,
    max_sequence_length,
    neg_max_sequence_length,
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

    try:
        steps = int(num_inference_steps)
        max_len     = int(max_sequence_length)
        neg_max_len = int(neg_max_sequence_length)
        n_images    = int(num_images_per_prompt)
        start_time  = time.time()

        prompt = (prompt or "").strip()
        if not prompt:
            return None, [], [], "오류: 프롬프트가 비어 있습니다."

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중...")
        print("=" * 60)
        print("[입력 프롬프트]")
        print(prompt)
        if negative and negative.strip():
            print("-" * 60)
            print("[네거티브 프롬프트]")
            print(negative)
        print("=" * 60)

        # ---- 토큰 수 확인 + 인코딩 (포지티브) ----
        raw_ids = pipe.tokenizer(
            prompt, truncation=False, add_special_tokens=True, return_tensors="pt"
        )["input_ids"][0]
        raw_token_count = len(raw_ids)
        clipped = max(0, raw_token_count - max_len)
        if clipped > 0:
            print(f"✗ 토큰 수: {raw_token_count} / {max_len} → {clipped}개 잘림!")
            try:
                truncated_text = pipe.tokenizer.decode(
                    raw_ids[max_len:].tolist(), skip_special_tokens=True
                )
                print("-" * 60)
                print("✗ [잘린 텍스트]")
                print(truncated_text)
                print("-" * 60)
            except Exception as e:
                print(f"✗ [잘린 텍스트 디코드 실패: {e}]")
        else:
            print(f"✓ 토큰 수: {raw_token_count} / {max_len} (잘림 없음)")

        # ---- PE (Prompt Enhancer) 적용 ----
        if use_pe and pipe.pe is not None and pipe.pe_tokenizer is not None:
            print("PE로 프롬프트 개선 중...")
            # model_cpu_offload 환경에서 PE는 CPU에 오프로드된 상태이므로
            # _execution_device(cuda)가 아닌 PE 모델의 실제 디바이스를 사용해야 함
            pe_device = next(pipe.pe.parameters()).device
            prompt = pipe._enhance_prompt_with_pe(prompt, pe_device, width=width, height=height)
            print("-" * 60)
            print("[PE 개선된 프롬프트]")
            print(prompt)
            print("-" * 60)

        prompt_embeds = _encode_prompt(prompt, max_len, n_images)

        # ---- 토큰 수 확인 + 인코딩 (네거티브) ----
        neg_token_count = 0
        neg_clipped = 0
        negative_prompt_embeds = None
        neg_text = (negative or "").strip()
        if neg_text:
            neg_raw_ids = pipe.tokenizer(
                neg_text, truncation=False, add_special_tokens=True, return_tensors="pt"
            )["input_ids"][0]
            neg_token_count = len(neg_raw_ids)
            neg_clipped = max(0, neg_token_count - neg_max_len)
            if neg_clipped > 0:
                print(f"✗ 네거티브 토큰 수: {neg_token_count} / {neg_max_len} → {neg_clipped}개 잘림!")
                try:
                    neg_truncated_text = pipe.tokenizer.decode(
                        neg_raw_ids[neg_max_len:].tolist(), skip_special_tokens=True
                    )
                    print("-" * 60)
                    print("✗ [잘린 네거티브 텍스트]")
                    print(neg_truncated_text)
                    print("-" * 60)
                except Exception as e:
                    print(f"✗ [잘린 네거티브 텍스트 디코드 실패: {e}]")
            else:
                print(f"✓ 네거티브 토큰 수: {neg_token_count} / {neg_max_len} (잘림 없음)")
            negative_prompt_embeds = _encode_prompt(neg_text, neg_max_len, n_images)
        else:
            # 네거티브 없을 때 빈 문자열로 인코딩 (CFG용)
            negative_prompt_embeds = _encode_prompt("", neg_max_len, n_images)

        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        progress(0.05, desc="추론 시작...")

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
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            # PE는 이미 수동 적용됨 → 파이프라인 내부 PE 비활성화
            "use_pe": False,
            # n_images는 embeds에 이미 반영 → 파이프라인에는 1로 전달
            "num_images_per_prompt": 1,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
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
            f"{script_name}_{timestamp}_{device_label}_{width}x{height}"
            f"_gs{guidance_scale}_step{steps}_seed{int(seed)}"
            f"_pe{int(use_pe)}_n{int(num_images_per_prompt)}"
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

        saved_info = (
            f"저장됨: {saved_files[0]}"
            if len(saved_files) == 1
            else f"{len(saved_files)}장 저장됨: {saved_files[0]} 외"
        )

        token_warn = clipped > 0 or neg_clipped > 0
        token_info = (
            f"토큰: {raw_token_count}/{max_len}"
            + (f" → {clipped}개 잘림!" if clipped > 0 else "")
            + (f" | 네거티브: {neg_token_count}/{neg_max_len}" if neg_token_count else "")
            + (f" → {neg_clipped}개 잘림!" if neg_clipped > 0 else "")
        )
        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초 | {token_info}")

        progress(1.0, desc="완료!")
        status_prefix = "⚠ 완료 (토큰 잘림)" if token_warn else "✓ 완료"
        return (
            make_image_grid(images),
            images,
            saved_files,
            f"{status_prefix} ({elapsed:.1f}초) | {token_info} | {saved_info}",
        )
    except Exception as e:
        return None, [], [], f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()
    load_model()

    with gr.Blocks(title="ERNIE Text-to-Image Generator") as interface:
        gr.Markdown("# ERNIE Text-to-Image Generator")
        gr.Markdown(
            f"AI를 사용하여 텍스트에서 이미지를 생성합니다."
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

                gr.Markdown(
                    "### 프롬프트 구성\n"
                    "- **프롬프트**: Subject → 포즈·의상·머리 → Setting → Lighting → Camera → Positive 순으로 결합됩니다.\n"
                    "- **Negative**: 네거티브 프롬프트는 별도로 인코딩됩니다.\n"
                    "- **Prompt Enhancer (PE)**: ERNIE의 프롬프트 자동 개선 기능입니다."
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
                        info="프롬프트 맨 앞에 배치됩니다.",
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
                        info="프롬프트 끝에 추가됩니다.",
                    )
                    negative_box = gr.Textbox(
                        label="19. 네거티브 프롬프트 (Negative)",
                        value=NEGATIVE,
                        lines=2,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="별도 네거티브 인코딩에 사용됩니다.",
                    )
                with gr.Accordion("결합된 프롬프트 (전체 섹션)", open=False):
                    prompt_combined = gr.Textbox(
                        label="결합 프롬프트",
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
                        lines=6,
                        interactive=True,
                        info="Subject → Face → Head → Body → 의상 → 장식 → Arm/Hand → Leg → Foot → Setting → Lighting → Camera → Positive 순으로 자동 결합됩니다.",
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
                        outputs=[prompt_combined],
                    )

            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                param_info = gr.Markdown(
                    "- **이미지 크기**: 16의 배수여야 합니다 (VAE scale factor = 16).\n"
                    "- **최대 시퀀스 길이**: 모델 로드 후 토크나이저 최대값이 표시됩니다."
                )
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=16,
                        value=1024,
                        info="이미지 너비 (픽셀). 반드시 16의 배수. 권장: 1024.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=16,
                        value=1024,
                        info="이미지 높이 (픽셀). 반드시 16의 배수. 권장: 1024.",
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=4.0,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. ERNIE 권장: 4.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. ERNIE 권장: 50",
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
                        label="포지티브 최대 시퀀스 길이",
                        minimum=64,
                        maximum=2048,
                        step=64,
                        value=1024,
                        info="포지티브 프롬프트 토큰 한도 (model_max_length = 2048). 초과 토큰은 잘림.",
                    )
                    neg_max_sequence_length = gr.Slider(
                        label="네거티브 최대 시퀀스 길이",
                        minimum=64,
                        maximum=2048,
                        step=64,
                        value=1024,
                        info="네거티브 프롬프트 토큰 한도 (model_max_length = 2048). 초과 토큰은 잘림.",
                    )
                with gr.Row():
                    use_pe = gr.Checkbox(
                        label="Prompt Enhancer (PE) 사용",
                        value=True,
                        info="ERNIE 내장 프롬프트 개선 기능. 활성화 시 프롬프트를 자동으로 보강합니다.",
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

        _model_outputs = [
            device_status,
            width, height,
            guidance_scale, num_inference_steps, num_images_per_prompt,
            use_pe, max_sequence_length, neg_max_sequence_length,
            param_info,
        ]
        load_model_btn.click(
            fn=load_model,
            inputs=[device_selector],
            outputs=_model_outputs,
        )
        device_selector.change(
            fn=load_model,
            inputs=[device_selector],
            outputs=_model_outputs,
        )
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_combined,
                negative_box,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                num_images_per_prompt,
                seed,
                use_pe,
                max_sequence_length,
                neg_max_sequence_length,
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
