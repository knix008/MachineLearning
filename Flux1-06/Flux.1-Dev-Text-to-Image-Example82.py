import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import re
import torch
import platform
from diffusers import FluxPipeline
from datetime import datetime
from PIL import Image
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# Default values for each prompt section
# CLIP ~77토큰(Subject+Positive). T5는 FLUX.1-dev 기준 최대 512토큰(diffusers 한도, 초과 시 잘림).
T5_MODEL_MAX_LENGTH = 512

SUBJECT = "High-quality photorealistic cosplay portrait, young Korean woman, soft idol look, standing in a bright living room, wearing white fishnet stockings."

FOOT = "Feet flat on the living room floor, balanced relaxed stance."

LEG = "Legs straight, soft knees, natural standing posture."

FACE = "Fair clear skin, vivid blue contact lenses, innocent curious gaze at camera. Long straight jet-black hair with thick straight-cut bangs framing her face."

BODY = "Upright in the living room, torso to camera, shoulders relaxed."

ARM = "Both arms hang naturally at her sides, elbows softly bent, relaxed."

HAND = "Both hands beside her thighs, fingers relaxed, empty hands."

FOOTWEAR = ""

LEGWEAR = "White fishnet stockings, blue-and-white ruffled lace garters with small white bows."

BOTTOM = ""

TOP = "Blue denim-textured bodysuit, front zipper, silver buttons, thin silver chains on chest, semi-sheer white lace side panels, blue bow tie on white collar, long white floral lace fingerless sleeves past elbows with blue cuffs and tiny black ribbon accents."

HEADWEAR = "Tall blue fabric bunny ears, white lace lining, white lace headband base, small white bow."

ARMWEAR = ""

HEAD = "Innocent curious expression, eyes to camera."

SETTING = "Tasteful living room: sofa, coffee table, decor, daylight windows; she stands on the floor; soft bokeh background."

LIGHTING = "Bright soft indoor and window light, even, flattering, minimal harsh shadows."

CAMERA = "Eye-level, full-length or three-quarter in living room, sharp on subject, soft room bokeh."

POSITIVE = "8k, photorealistic, sharp, perfect anatomy, ten fingers, well-formed hands, no extra fingers, five finger for each hands."

NEGATIVE = "Blurry, low quality, distorted, deformed, ugly, bad anatomy, bad hands, bad fingers, extra fingers, missing fingers, extra limbs, missing limbs, fused fingers, too many fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, extra arms, extra legs, malformed limbs, watermark, text, signature, logo, jpeg artifacts, cropped, worst quality, normal quality"


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


DEFAULT_COMBINED_T5_PROMPT = combine_t5_prompt_sections(
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
            f"T5 토큰 약 {n}{est} / 한도 {max_len} — 약 {over}토큰이 잘립니다. "
            "섹션을 짧게 하거나 T5 박스에서 직접 줄이세요."
        )
    return f"T5 토큰 약 {n}{est} / 한도 {max_len} (이 길이에서는 잘림 없음)."


def combine_prompt_sections_dual_with_stats(
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
    clip_prompt, t5_prompt = combine_prompt_sections_dual(
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
    return clip_prompt, t5_prompt, format_t5_length_hint(t5_prompt)


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
    """Build CLIP and T5 prompt strings for FLUX.1 dual encoders."""
    clip_prompt = format_clip_preview(subject, positive)
    t5_prompt = combine_t5_prompt_sections(
        subject, foot, leg, face, body, arm, hand,
        footwear, legwear, bottom, top, headwear, armwear,
        head, setting, lighting, camera, positive,
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
    """Load and initialize the Flux model with optimizations."""
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
    print("CLIP + T5 듀얼 텍스트 인코더를 사용합니다.")
    pipe = FluxPipeline.from_pretrained(
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
        print("메모리 최적화 적용: attention slicing, VAE slicing, VAE tiling (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt_clip,
    prompt_t5,
    negative,
    width,
    height,
    guidance_scale,
    true_cfg_scale,
    num_inference_steps,
    num_images_per_prompt,
    seed,
    max_sequence_length,
    image_format,
    progress=gr.Progress(track_tqdm=True),
):
    global pipe

    if pipe is None:
        return (
            None, [], [],
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        prompt_clip = (prompt_clip or "").strip()
        if not prompt_clip:
            return None, [], [], "오류: CLIP 프롬프트(미리보기)가 비어 있습니다."

        prompt_t5 = (prompt_t5 or "").strip()
        if not prompt_t5:
            return None, [], [], "오류: T5 프롬프트(prompt_2)를 입력해주세요."

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중...")
        print("=" * 60)
        print("[입력 CLIP 프롬프트]")
        print(prompt_clip)
        print("-" * 60)
        print("[입력 T5 프롬프트]")
        print(prompt_t5)
        print("=" * 60)

        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # ---- Encode CLIP ----
        clip_max_len = getattr(pipe, "tokenizer_max_length", None)
        if (
            clip_max_len is None
            and hasattr(pipe, "tokenizer")
            and pipe.tokenizer is not None
        ):
            clip_max_len = getattr(pipe.tokenizer, "model_max_length", None)
        clip_max_len = int(clip_max_len) if clip_max_len is not None else 77

        clip_inputs = pipe.tokenizer(
            prompt_clip,
            padding="max_length",
            max_length=clip_max_len,
            truncation=True,
            return_tensors="pt",
        )

        raw_clip_ids = pipe.tokenizer(
            prompt_clip, truncation=False, return_tensors="pt"
        )["input_ids"][0]
        raw_clip_token_count = len(raw_clip_ids)
        clipped_clip = max(0, raw_clip_token_count - clip_max_len)
        if clipped_clip > 0:
            print(f"✗ CLIP 토큰 수: {raw_clip_token_count} / {clip_max_len} → {clipped_clip}개 잘림!")
            try:
                tail_ids = raw_clip_ids[clip_max_len:]
                if hasattr(tail_ids, "tolist"):
                    tail_ids = tail_ids.tolist()
                truncated_clip_text = pipe.tokenizer.decode(tail_ids, skip_special_tokens=True)
                print("-" * 60)
                print("✗ [잘린 CLIP 텍스트]")
                print(truncated_clip_text)
                print("-" * 60)
            except Exception as e:
                print(f"✗ [잘린 CLIP 텍스트 디코드 실패: {e}]")
        else:
            print(f"✓ CLIP 토큰 수: {raw_clip_token_count} / {clip_max_len} (잘림 없음)")

        with torch.inference_mode():
            clip_out = pipe.text_encoder(
                clip_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )
        pooled_prompt_embeds = clip_out.pooler_output.to(dtype=DTYPE)

        # ---- Encode T5 ----
        max_len = min(int(max_sequence_length), T5_MODEL_MAX_LENGTH)
        text_inputs = pipe.tokenizer_2(
            prompt_t5,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )

        raw_ids = pipe.tokenizer_2(prompt_t5, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
        raw_token_count = len(raw_ids)
        clipped = max(0, raw_token_count - max_len)
        if clipped > 0:
            print(f"✗ T5 토큰 수: {raw_token_count} / {max_len} → {clipped}개 잘림!")
            truncated_text = pipe.tokenizer_2.decode(raw_ids[max_len:], skip_special_tokens=True)
            print("-" * 60)
            print("✗ [잘린 T5 텍스트]")
            print(truncated_text)
            print("-" * 60)
        else:
            print(f"✓ T5 토큰 수: {raw_token_count} / {max_len} (잘림 없음)")

        with torch.inference_mode():
            prompt_embeds = pipe.text_encoder_2(
                text_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=DTYPE)

        # Encode negative prompt when true_cfg_scale > 1.0
        negative_t5_embeds = None
        if true_cfg_scale > 1.0 and negative and negative.strip():
            print(f"네거티브 프롬프트 인코딩 중... (true_cfg_scale={true_cfg_scale})")
            neg_clip_text = negative
            neg_t5_text = negative

            negative_clip_pooled_prompt_embeds = None
            neg_clip_inputs = pipe.tokenizer(
                neg_clip_text,
                padding="max_length",
                max_length=clip_max_len,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                neg_clip_out = pipe.text_encoder(
                    neg_clip_inputs["input_ids"].to(DEVICE),
                    output_hidden_states=False,
                )
            negative_clip_pooled_prompt_embeds = neg_clip_out.pooler_output.to(dtype=DTYPE)

            neg_inputs = pipe.tokenizer_2(
                neg_t5_text,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                negative_t5_embeds = pipe.text_encoder_2(
                    neg_inputs["input_ids"].to(DEVICE),
                    output_hidden_states=False,
                )[0]
            negative_t5_embeds = negative_t5_embeds.to(dtype=DTYPE)

        progress(0.05, desc="추론 시작...")

        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.90
            progress(progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)")
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
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": steps,
            "num_images_per_prompt": int(num_images_per_prompt),
            "generator": generator,
            "callback_on_step_end": step_callback,
        }
        if negative_t5_embeds is not None:
            pipe_kwargs["negative_prompt_embeds"] = negative_t5_embeds
            pipe_kwargs["negative_pooled_prompt_embeds"] = negative_clip_pooled_prompt_embeds
            pipe_kwargs["true_cfg_scale"] = true_cfg_scale

        with torch.inference_mode():
            images = pipe(**pipe_kwargs).images

        progress(0.95, desc="이미지 저장 중...")

        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        if DEVICE == "cuda" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
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
            f"_cfg{true_cfg_scale}_n{int(num_images_per_prompt)}_msl{max_len}"
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
            f"T5 토큰: {raw_token_count}/{max_len} → {clipped}개 잘림! | "
            f"CLIP 토큰: {raw_clip_token_count}/{clip_max_len} → {clipped_clip}개 잘림!"
            if clipped > 0 or clipped_clip > 0
            else f"T5 토큰: {raw_token_count}/{max_len} | CLIP 토큰: {raw_clip_token_count}/{clip_max_len}"
        )
        saved_info = (
            f"저장됨: {saved_files[0]}"
            if len(saved_files) == 1
            else f"{len(saved_files)}장 저장됨: {saved_files[0]} 외"
        )
        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초 | {token_info}")

        progress(1.0, desc="완료!")
        return (
            make_image_grid(images),
            images,
            saved_files,
            f"✓ 완료! ({elapsed:.1f}초) | {token_info} | {saved_info}",
        )
    except Exception as e:
        return None, [], [], f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()
    load_model()

    with gr.Blocks(title="Flux.1-dev Text-to-Image Generator") as interface:
        gr.Markdown("# Flux.1-dev Text-to-Image Generator")
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
                    "- **CLIP**: Subject → Positive 순. Positive는 T5에 포함되지 않습니다.\n"
                    "- **T5**: Subject → 포즈·의상·머리 → Setting → Lighting → Camera 순.\n"
                    f"- **T5 한도**: FLUX.1-dev는 최대 **{T5_MODEL_MAX_LENGTH}토큰**(초과분 잘림).\n"
                    "- **Negative**: True CFG > 1.0일 때 CLIP/T5 공통 적용."
                )
                with gr.Accordion("프롬프트 섹션 (Subject · 포즈 · 의상 · 배경 · 조명 · 카메라)", open=True):
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
                        info="T5 프롬프트에 포함됩니다. CLIP에는 포함되지 않습니다.",
                    )
                    negative_box = gr.Textbox(
                        label="19. 네거티브 프롬프트 (Negative)",
                        value=NEGATIVE,
                        lines=2,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="True CFG Scale > 1.0일 때 CLIP/T5 모두에 공통으로 사용됩니다.",
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
                        value=DEFAULT_COMBINED_T5_PROMPT,
                        lines=5,
                        interactive=True,
                        info=(
                            "Subject → 포즈·의상·머리 → Setting → Lighting → Camera → Positive 순으로 자동 결합. "
                            f"T5는 최대 약 {T5_MODEL_MAX_LENGTH}토큰까지 반영됩니다."
                        ),
                    )
                    t5_length_hint = gr.Textbox(
                        label="T5 길이",
                        value=format_t5_length_hint(DEFAULT_COMBINED_T5_PROMPT),
                        lines=2,
                        interactive=False,
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
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256, maximum=2048, step=32, value=768,
                        info="이미지 너비 (픽셀). 32의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256, maximum=2048, step=32, value=1536,
                        info="이미지 높이 (픽셀). 32의 배수.",
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0, maximum=10.0, step=0.5, value=7.5,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. Flux.1 Dev 권장: 3.5~8",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10, maximum=50, step=1, value=28,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 25-40",
                    )
                with gr.Row():
                    true_cfg_scale = gr.Slider(
                        label="True CFG Scale (네거티브 프롬프트 강도)",
                        minimum=1.0, maximum=5.0, step=0.5, value=1.5,
                        info="1.0이면 네거티브 프롬프트 비활성화. 1.5~2.0 권장.",
                    )
                    num_images_per_prompt = gr.Slider(
                        label="생성 이미지 수",
                        minimum=1, maximum=4, step=1, value=1,
                        info="한 번에 생성할 이미지 수. 많을수록 VRAM 사용 증가.",
                    )
                with gr.Row():
                    seed = gr.Number(
                        label="시드", value=42, precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                    max_sequence_length = gr.Slider(
                        label="최대 시퀀스 길이 (T5)",
                        minimum=64,
                        maximum=T5_MODEL_MAX_LENGTH,
                        step=64,
                        value=T5_MODEL_MAX_LENGTH,
                        info=(
                            f"T5 인코딩 상한(최대 {T5_MODEL_MAX_LENGTH}, FLUX.1-dev 한도). "
                            "더 긴 프롬프트는 잘립니다."
                        ),
                    )
                with gr.Row():
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
                prompt_clip, prompt_t5, negative_box,
                width, height, guidance_scale, true_cfg_scale,
                num_inference_steps, num_images_per_prompt,
                seed, max_sequence_length, image_format,
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
