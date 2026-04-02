import re
import torch
import platform
from diffusers import FluxPipeline
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
SUBJECT = "A full body photograph of a beautiful slim young Korean woman with soft idol aesthetics, standing beside an indoor swimming pool, wearing a very small white bikini swimsuit, body facing almost directly forward with only a very slight side turn, with bare feet."

FOOT = "Bare feet fully visible and unobstructed on the dry poolside floor, including toes and both ankles; one foot bearing most weight, the other lightly touching the ground with relaxed toes."

LEG = "Bare legs very long in proportion to the body, slender and clearly elongated from hips to ankles, strong leggy silhouette, one leg straight and extended, the other slightly bent at the knee in a relaxed asymmetric stance."

FACE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, mouth closed with lips lightly pressed together, looking directly at the camera. She has long, voluminous wavy jet-black hair naturally flowing down."

BODY = "Torso and hips facing almost frontally toward the camera with only a subtle side angle, one shoulder only slightly closer to the lens, subtle S-curve pose, slender hourglass silhouette with clearly defined waist and clearly visible natural body curves."

ARM = "Both arms relaxed in a natural pose, one arm softly bent near the waist and the other resting comfortably along her side."

HAND = "Hands relaxed with naturally curved fingers, one hand resting near the waist and the other near the thigh."

FOOTWEAR = ""

LEGWEAR = ""

BOTTOM = "Very tiny white thong bikini swimsuit with a clean minimal design and smooth stretch fabric with minimal coverage."

TOP = "Very tiny white triangle bikini top with a clean minimal design and smooth stretch fabric with minimal coverage."

HEADWEAR = ""

ARMWEAR = ""

HEAD = "Head slightly tilted."

LIGHTING = "Strong frontal lighting directly from the front, even illumination across the whole body, bright and clean exposure, skin luminous and glowing."

SETTING = "Indoor luxury swimming poolside, clear blue pool water nearby, bright clean interior lighting, polished floor, soft reflections on the water surface."

CAMERA = "35mm lens, entire figure from head to feet in frame with both feet clearly visible near the bottom of the frame, camera positioned at waist level, slight upward perspective to lengthen the legs, tack sharp focus."

POSITIVE = "8k, high quality, photorealistic, perfect anatomy, ten fingers, natural skin texture, ten toes, beautiful toes, ultra sharp details."

NEGATIVE = "Blurry, soft focus, hazy, low sharpness, grainy, low quality, deformed, bad anatomy, extra limbs, ugly, watermark, text, signature, extra fingers, extra toes, deformed hands, high angle shot, overhead shot, bird's-eye view, anime, manga, cartoon, comics, illustration, anime style, cel shading, lineart, 2d, painterly, ghibli"


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
    # Ensure single space after comma, period, colon, semicolon
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    # Collapse multiple spaces into one
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
    lighting,
    setting,
    camera,
):
    """Combine separate prompt sections into one final prompt string."""
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
        lighting,
        setting,
        camera,
    ]
    # Filter out empty sections and join with a space, preserving original punctuation
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
    lighting,
    setting,
    camera,
):
    """
    Build two prompt strings for FLUX.1 dual encoders.

    - First is used as CLIP prompt (`prompt`)
    - Second is used as T5 prompt (`prompt_2`)
    """
    # CLIP(prompt): subject만 담당하고, positive는 generate_image에서 추가합니다.
    clip_prompt = normalize_spacing(subject) if subject else ""

    # T5(prompt_2): subject 제외 + 나머지 섹션 담당합니다.
    t5_prompt = combine_prompt_sections(
        "",
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
        lighting,
        setting,
        camera,
    )
    return clip_prompt, t5_prompt


def combine_prompt_sections_clip_only(
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
    lighting,
    setting,
    camera,
):
    """CLIP용 프롬프트만 생성합니다. (T5는 subject를 포함하지 않게 분리)"""
    clip_prompt, _t5_prompt = combine_prompt_sections_dual(
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
        lighting,
        setting,
        camera,
    )
    return clip_prompt


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
VERBOSE_CLI = False  # True로 바꾸면 프롬프트/토큰/진행 로그가 CLI에 출력됩니다.
CLI_PROGRESS = True  # 진행률(스텝 진행 바)만 CLI에 출력할지 여부


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

    # OS / Python / PyTorch
    print(
        f"OS: {platform.system()} {platform.release()} ({platform.machine()}) | "
        f"Python: {platform.python_version()} | PyTorch: {torch.__version__}"
    )

    # CPU
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_physical} physical / {cpu_logical} logical cores")

    # RAM
    mem = psutil.virtual_memory()
    print(
        f"RAM: {mem.total / (1024**3):.1f} GB (available {mem.available / (1024**3):.1f} GB)"
    )

    # GPU / MPS / CPU
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
    if VERBOSE_CLI:
        print("\n리소스 정리 중...")

    if interface is not None:
        try:
            interface.close()
            if VERBOSE_CLI:
                print("Gradio 서버 종료됨")
        except Exception:
            pass
        interface = None

    if pipe is not None:
        del pipe
        pipe = None
        if VERBOSE_CLI:
            print("모델 해제됨")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if VERBOSE_CLI:
            print("CUDA 캐시 정리됨")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        if VERBOSE_CLI:
            print("MPS 캐시 정리됨")

    gc.collect()
    if VERBOSE_CLI:
        print("메모리 정리 완료")


def signal_handler(_sig, _frame):
    """Handle keyboard interrupt signal."""
    global VERBOSE_CLI
    prev_verbose = VERBOSE_CLI
    # CTRL+C 상황에서는 원래처럼 메시지를 표시합니다.
    VERBOSE_CLI = True
    print("\n\n키보드 인터럽트 감지됨 (Ctrl+C)")
    cleanup()
    VERBOSE_CLI = prev_verbose
    sys.exit(0)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def load_model(device_name=None):
    """Load and initialize the Flux model with optimizations."""
    global pipe, DEVICE, DTYPE

    if device_name is not None:
        DEVICE = device_name
        DTYPE = torch.bfloat16 if device_name in ("cuda", "mps") else torch.float32

    # Release previous model if loaded
    if pipe is not None:
        if VERBOSE_CLI:
            print("기존 모델 해제 중...")
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if VERBOSE_CLI:
        print(f"모델 로딩 중... (Device: {DEVICE}, dtype: {DTYPE})")
        print("CLIP + T5 듀얼 텍스트 인코더를 사용합니다.")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    # Enable memory optimizations based on device
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        if VERBOSE_CLI:
            print(
                "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CUDA)"
            )
    elif DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        if VERBOSE_CLI:
            print(
                "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing (CPU)"
            )
    elif DEVICE == "mps":
        pipe.enable_attention_slicing()
        # channels_last memory format for better MPS performance
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)
        elif hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
        if VERBOSE_CLI:
            print(
                "메모리 최적화 적용: attention slicing, VAE slicing, VAE tiling (MPS)"
            )

    if VERBOSE_CLI:
        print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt_clip,
    prompt_t5,
    positive,
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
            None,
            [],
            [],
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    if not prompt_clip or not prompt_t5:
        return (
            None,
            [],
            [],
            "오류: CLIP 프롬프트(prompt)와 T5 프롬프트(prompt_2)를 입력해주세요.",
        )

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        # Append positive prompt to both CLIP and T5 prompts.
        if positive and positive.strip():
            prompt_clip = prompt_clip.rstrip() + " " + positive.strip()
            prompt_t5 = prompt_t5.rstrip() + " " + positive.strip()

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중...")
        print("=" * 60)
        print("[입력 CLIP 프롬프트]")
        print(prompt_clip)
        print("-" * 60)
        print("[입력 T5 프롬프트]")
        print(prompt_t5)
        print("=" * 60)

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # ---- Encode CLIP (pooled_prompt_embeds) ----
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

        # Count tokens to detect clipping
        raw_clip_ids = pipe.tokenizer(
            prompt_clip, truncation=False, return_tensors="pt"
        )["input_ids"][0]
        raw_clip_token_count = len(raw_clip_ids)
        clipped_clip = max(0, raw_clip_token_count - clip_max_len)
        if clipped_clip > 0:
            print(
                f"✗ CLIP 토큰 수: {raw_clip_token_count} / {clip_max_len} → {clipped_clip}개 잘림!"
            )
            # CLIP 잘린(초과) 부분을 텍스트로도 표시
            try:
                tail_ids = raw_clip_ids[clip_max_len:]
                if hasattr(tail_ids, "tolist"):
                    tail_ids = tail_ids.tolist()
                truncated_clip_text = pipe.tokenizer.decode(
                    tail_ids, skip_special_tokens=True
                )
                print("-" * 60)
                print("✗ [잘린 CLIP 텍스트]")
                print(truncated_clip_text)
                print("-" * 60)
            except Exception as e:
                print(f"✗ [잘린 CLIP 텍스트 디코드 실패: {e}]")
        else:
            print(
                f"✓ CLIP 토큰 수: {raw_clip_token_count} / {clip_max_len} (잘림 없음)"
            )

        with torch.inference_mode():
            clip_out = pipe.text_encoder(
                clip_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )
        pooled_prompt_embeds = clip_out.pooler_output.to(dtype=DTYPE)

        # ---- Encode T5 (prompt_embeds) ----
        max_len = int(max_sequence_length)
        text_inputs = pipe.tokenizer_2(
            prompt_t5,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )

        # Count tokens to detect clipping
        raw_ids = pipe.tokenizer_2(prompt_t5, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
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

        with torch.inference_mode():
            prompt_embeds = pipe.text_encoder_2(
                text_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=DTYPE)

        # Encode negative prompt when true_cfg_scale > 1.0
        # - T5 embeddings 변수는 여기서 기본값을 미리 선언합니다.
        # - CLIP pooled 변수는 아래 CLIP negative 인코딩 블록에서 선언합니다. (요청: 선언 위치 분리)
        negative_t5_embeds = None
        if true_cfg_scale > 1.0 and negative and negative.strip():
            if VERBOSE_CLI:
                print(
                    f"네거티브 프롬프트 인코딩 중... (true_cfg_scale={true_cfg_scale})"
                )
            neg_clip_text = negative
            neg_t5_text = negative

            # Negative CLIP pooled embedding
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
            negative_clip_pooled_prompt_embeds = neg_clip_out.pooler_output.to(
                dtype=DTYPE
            )

            # Negative T5 token embeddings
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

        # Callback to report each inference step to Gradio progress bar and CLI status bar
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.90
            progress(
                progress_val,
                desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)",
            )

            # CLI status bar
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

        # Build pipeline kwargs with pre-computed T5 embeddings
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
            pipe_kwargs["negative_pooled_prompt_embeds"] = (
                negative_clip_pooled_prompt_embeds
            )
            pipe_kwargs["true_cfg_scale"] = true_cfg_scale

        # Run the pipeline
        with torch.inference_mode():
            images = pipe(**pipe_kwargs).images

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
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
            f"_cfg{true_cfg_scale}_n{int(num_images_per_prompt)}_msl{int(max_sequence_length)}"
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

    # Print hardware specifications
    print_hardware_info()

    # Auto-load model on startup with detected device
    load_model()

    # Create Gradio interface
    with gr.Blocks(
        title="Flux.1-dev Text-to-Image Generator",
    ) as interface:
        gr.Markdown("# Flux.1-dev Text-to-Image Generator")
        gr.Markdown(
            f"AI를 사용하여 텍스트에서 이미지를 생성합니다."
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
                prompt_subject = gr.Textbox(
                    label="1. 주제/대상 (Subject)",
                    value=SUBJECT,
                    lines=2,
                    placeholder="예: 1girl, young woman, a cat",
                    info="이미지의 주된 주제나 대상을 설명합니다.",
                )
                prompt_foot = gr.Textbox(
                    label="2. 포즈 - 발 (Pose: Foot)",
                    value=FOOT,
                    lines=1,
                    placeholder="예: feet slightly apart, toes pointed forward",
                    info="발의 위치를 설명합니다.",
                )
                prompt_leg = gr.Textbox(
                    label="3. 포즈 - 다리 (Pose: Leg)",
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
                    label="5. 포즈 - 몸통 (Pose: Body)",
                    value=BODY,
                    lines=2,
                    placeholder="예: body angled slightly, leaning forward",
                    info="몸통 자세와 전체 실루엣을 설명합니다.",
                )
                prompt_arm = gr.Textbox(
                    label="6. 포즈 - 팔 (Pose: Arm)",
                    value=ARM,
                    lines=1,
                    placeholder="예: arms resting across torso",
                    info="팔의 위치와 자세를 설명합니다.",
                )
                prompt_hand = gr.Textbox(
                    label="7. 포즈 - 손 (Pose: Hand)",
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
                    label="14. 포즈 - 머리 (Pose: Head)",
                    value=HEAD,
                    lines=1,
                    placeholder="예: head tilted slightly, gazing off-camera",
                    info="머리와 시선 방향을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="15. 조명 (Lighting)",
                    value=LIGHTING,
                    lines=2,
                    placeholder="예: golden hour, city glow, cinematic rim light",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="16. 배경/장소 (Setting & Background)",
                    value=SETTING,
                    lines=2,
                    placeholder="예: rooftop terrace at twilight, city skyline",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="17. 카메라 설정 (Camera Settings)",
                    value=CAMERA,
                    lines=2,
                    placeholder="예: Sony A7R V, 85mm f/1.8, ISO 400",
                    info="카메라 기종, 렌즈, ISO, 셔터 스피드, 조리개, 피사계 심도 등을 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Dual Encoders)", open=False):
                    prompt_clip = gr.Textbox(
                        label="CLIP 프롬프트 (prompt)",
                        value=SUBJECT,
                        lines=4,
                        interactive=True,
                        info="CLIP용 프롬프트입니다. 기본값은 `subject`만 들어가며, `positive`는 생성 시 여기에 함께 추가됩니다. (negative는 별도 박스에서 처리)",
                    )
                    prompt_t5 = gr.Textbox(
                        label="T5 프롬프트 (prompt_2)",
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
                            LIGHTING,
                            SETTING,
                            CAMERA,
                        ),
                        lines=4,
                        interactive=True,
                        info="T5용 프롬프트입니다. 기본값은 `subject`를 제외한 나머지 섹션이고, `positive`는 생성 시 여기에 함께 추가됩니다. (T5 토큰은 max_sequence_length까지)",
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
                    prompt_lighting,
                    prompt_setting,
                    prompt_camera,
                ]
                for section in prompt_sections:
                    section.change(
                        fn=combine_prompt_sections_dual,
                        inputs=prompt_sections,
                        outputs=[prompt_clip, prompt_t5],
                    )

                with gr.Accordion("포지티브/네거티브 프롬프트", open=False):
                    positive_box = gr.Textbox(
                        label="포지티브 프롬프트 (Positive Prompt)",
                        value=POSITIVE,
                        lines=2,
                        placeholder="예: masterpiece, best quality, highly detailed",
                        info="최종 프롬프트 뒤에 추가로 덧붙일 키워드를 입력합니다.",
                    )
                    negative_box = gr.Textbox(
                        label="네거티브 프롬프트 (Negative Prompt)",
                        value=NEGATIVE,
                        lines=2,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="True CFG Scale > 1.0일 때 CLIP/T5 모두에 공통으로 사용됩니다.",
                    )

            # Right column: Parameters (top) + Image generation (bottom)
            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=32,
                        value=768,
                        info="이미지 너비 (픽셀). 32의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=32,
                        value=1536,
                        info="이미지 높이 (픽셀). 32의 배수.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=7.5,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. Flux.1 Dev 권장: 3.5~8",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 25-40",
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
                prompt_clip,
                prompt_t5,
                positive_box,
                negative_box,
                width,
                height,
                guidance_scale,
                true_cfg_scale,
                num_inference_steps,
                num_images_per_prompt,
                seed,
                max_sequence_length,
                image_format,
            ],
            outputs=[output_grid, output_gallery, output_files, output_message],
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
