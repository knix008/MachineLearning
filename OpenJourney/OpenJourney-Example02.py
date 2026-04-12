import re
import torch
import platform
from diffusers import DiffusionPipeline
import diffusers
import transformers
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# Suppress position_ids unexpected key warnings from model loading
transformers.logging.set_verbosity_error()
diffusers.utils.logging.set_verbosity_error()

# Default values for each prompt section
SUBJECT = "A full body photography of a beautiful young skinny Korean woman with soft idol aesthetics standing on a casual spring outing in Seoul, wearing black stiletto high heels and a black bodycon mini dress, mdjrny-v4 style."

FOOT = "Both feet pressed firmly together, inner edges of both shoes touching, heels touching, no gap between feet, feet perfectly parallel side by side, entire shoes fully visible from toe to heel tip, generous empty space below the heel tips."

LEG = "Both legs fully straight and pressed firmly together, knees touching each other, inner thighs touching, no gap between legs from thigh to ankle, legs completely closed, not crossed."

FACE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark black hair. Her expression is innocent and curious with a very subtle soft smile, mouth gently closed, lips together, looking directly at the camera. She has long wavy voluminous jet-black hair with beautiful soft waves and curls, dramatically flowing and billowing in the wind, strands sweeping through the air."

BODY = "Weight shifted onto one straight supporting leg with a subtle hip shift, creating a soft S-curve; both legs straight and together; body facing completely straight toward the camera, chest and torso fully frontal, posture tall and elegant."

ARM = "Both arms hanging naturally and relaxed at her sides."

HAND = "Both hands hanging gracefully at her sides."

FOOTWEAR = "Black pointed-toe stiletto pumps, thin spike heels clearly visible, both shoes entirely visible from toe to heel tip, spike heels fully shown touching the ground."

LEGWEAR = ""

BOTTOM = ""

TOP = "Sleeveless black bodycon mini dress with a deep plunging V-neckline, very short hemline well above mid-thigh, extremely tight fit hugging the waist and hips, solid black fabric stretching tightly over her slim figure, small side slit on one side of the skirt revealing a glimpse of the leg."

HEADWEAR = ""

ARMWEAR = "Delicate gold chain necklace clearly visible at the neckline, delicate gold chain bracelet on one wrist."

HEAD = "Head tilted just slightly to one side at a very gentle angle, soft and relaxed posture."

SETTING = "Bright spring street in Seoul, cherry blossom trees lining the sidewalk with pink petals falling gently, warm sunny day, clean pavement."

LIGHTING = "Bright even spring daylight, soft frontal natural light, face clearly and brightly lit, no harsh shadows."

CAMERA = "Full body shot, head to heel tips fully in frame, extra space above head and below heels, ankle level angle, sharp focus, soft bokeh background."

POSITIVE = "8k, high quality, realistic, perfect anatomy, ten fingers, detailed high heels fully visible, stiletto spike heels clearly shown, full body visible, legs and feet in frame."

NEGATIVE = "Blurry, low quality, deformed, bad anatomy, extra limbs, ugly, watermark, text, signature, extra fingers, one leg forward, staggered legs, walking pose, legs apart, feet apart, gap between feet, gap between knees, gap between legs, space between legs, open legs, wide stance, crossed legs, equal weight on both legs, flat silhouette, no hip shift, heels cropped, heel tips hidden, heels not visible, feet cropped, missing feet, missing shoes, shoes cropped, bikini, swimsuit."


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
    """Combine all prompt sections into a single prompt string."""
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


def get_device_and_dtype():
    """Detect the best available device and appropriate data type."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
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
    """Load and initialize the OpenJourney model with optimizations."""
    global pipe, DEVICE, DTYPE

    if device_name is not None:
        DEVICE = device_name
        DTYPE = torch.float16 if device_name in ("cuda", "mps") else torch.float32

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
    pipe = DiffusionPipeline.from_pretrained(
        "prompthero/openjourney",
        custom_pipeline="lpw_stable_diffusion",
        torch_dtype=DTYPE,
    )
    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (CUDA)")
    elif DEVICE == "cpu":
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (CPU)")
    elif DEVICE == "mps":
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: attention slicing (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt,
    negative,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    num_images_per_prompt,
    seed,
    max_embeddings_multiples,
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

        prompt = (prompt or "").strip()
        if not prompt:
            return None, [], [], "오류: 프롬프트가 비어 있습니다."

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중...")
        print("=" * 60)
        print("[입력 프롬프트]")
        print(prompt)
        print("=" * 60)

        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

        progress(0.05, desc="추론 시작...")

        def step_callback(step_index, _timestep, _latents):
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

        with torch.inference_mode():
            images = pipe(
                prompt=prompt,
                negative_prompt=negative if negative and negative.strip() else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                num_images_per_prompt=int(num_images_per_prompt),
                generator=generator,
                max_embeddings_multiples=int(max_embeddings_multiples),
                callback=step_callback,
                callback_steps=1,
            ).images

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
            f"_n{int(num_images_per_prompt)}"
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
    load_model()

    with gr.Blocks(title="OpenJourney Text-to-Image Generator") as interface:
        gr.Markdown("# OpenJourney Text-to-Image Generator")
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
                    "- 모든 섹션이 순서대로 결합되어 단일 프롬프트를 구성합니다.\n"
                    "- OpenJourney 트리거 워드 `mdjrny-v4 style`을 Subject에 포함하세요."
                )
                with gr.Accordion("프롬프트 섹션 (Subject · 포즈 · 의상 · 배경 · 조명 · 카메라)", open=True):
                    prompt_subject = gr.Textbox(
                        label="1. 주제/대상 (Subject)",
                        value=SUBJECT,
                        lines=2,
                        placeholder="예: 1girl, young woman, mdjrny-v4 style",
                        info="프롬프트 맨 앞. mdjrny-v4 style 트리거 워드를 포함하세요.",
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
                        placeholder="예: full body shot, sharp focus, soft bokeh",
                        info="앵글, 구도, 초점 등을 설명합니다.",
                    )
                with gr.Accordion("포지티브 프롬프트 (Positive)", open=False):
                    positive_box = gr.Textbox(
                        label="포지티브 프롬프트 (Positive Prompt)",
                        value=POSITIVE,
                        lines=2,
                        placeholder="예: masterpiece, best quality, highly detailed",
                        info="품질 및 세부 묘사 키워드.",
                    )
                with gr.Accordion("결합된 프롬프트 (전체 섹션)", open=False):
                    prompt_combined = gr.Textbox(
                        label="결합된 프롬프트",
                        value=combine_prompt_sections(
                            SUBJECT, FOOT, LEG, FACE, BODY, ARM, HAND,
                            FOOTWEAR, LEGWEAR, BOTTOM, TOP, HEADWEAR, ARMWEAR,
                            HEAD, SETTING, LIGHTING, CAMERA, POSITIVE,
                        ),
                        lines=5,
                        interactive=True,
                        info="모든 섹션이 순서대로 결합된 최종 프롬프트입니다.",
                    )
                with gr.Accordion("네거티브 프롬프트 (Negative)", open=False):
                    negative_box = gr.Textbox(
                        label="네거티브 프롬프트 (Negative Prompt)",
                        value=NEGATIVE,
                        lines=2,
                        placeholder="예: blurry, deformed hands, bad anatomy",
                        info="생성 시 제외할 요소.",
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
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256, maximum=1024, step=64, value=512,
                        info="이미지 너비 (픽셀). 64의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256, maximum=1024, step=64, value=768,
                        info="이미지 높이 (픽셀). 64의 배수.",
                    )
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0, maximum=20.0, step=0.5, value=7.5,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. 권장: 7~12",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10, maximum=100, step=1, value=50,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 30~50",
                    )
                with gr.Row():
                    num_images_per_prompt = gr.Slider(
                        label="생성 이미지 수",
                        minimum=1, maximum=4, step=1, value=1,
                        info="한 번에 생성할 이미지 수. 많을수록 VRAM 사용 증가.",
                    )
                    seed = gr.Number(
                        label="시드", value=42, precision=0,
                        info="난수 시드. 같은 값이면 같은 결과.",
                    )
                with gr.Row():
                    max_embeddings_multiples = gr.Slider(
                        label="최대 임베딩 배수 (Max Embeddings Multiples)",
                        minimum=1, maximum=16, step=1, value=14,
                        info="77토큰 청크 수. 14 = 최대 1078토큰. 높을수록 긴 프롬프트 허용.",
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
                prompt_combined, negative_box,
                width, height, guidance_scale,
                num_inference_steps, num_images_per_prompt,
                seed, max_embeddings_multiples, image_format,
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
