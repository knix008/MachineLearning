import re
import torch
import platform
from diffusers import DiffusionPipeline
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
DEFAULT_SUBJECT = "A full body photography of a beautiful young skinny Korean woman sitting at a café table, holding a Starbucks coffee cup."

DEFAULT_FACE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is gentle with a soft closed-mouth smile, lips lightly pressed together with a subtle upward curve, looking directly at the camera. She has long, voluminous straight jet-black hair with beautiful soft waves and curls, dramatically flowing and billowing in the wind, strands sweeping through the air."

DEFAULT_POSE_HEAD = "Head held upright with elegant posture, hair draping naturally over shoulders."

DEFAULT_HEADWEAR = ""

DEFAULT_POSE_LEG = "Both legs close together, slightly angled to the side, skirt side slit slightly parted open, bare leg clearly visible from mid-thigh downward through the slit."

DEFAULT_LEGWEAR = "Bare legs, smooth and fair skin, leg clearly exposed through the deep side slit of the skirt from mid-thigh to ankle."

DEFAULT_POSE_FOOT = "Both feet side by side, parallel to each other, toes pointing forward, white sneakers clearly shown side by side, feet fully in frame."

DEFAULT_FOOTWEAR = "Clean white canvas sneakers."

DEFAULT_POSE_ARM = "One arm resting on the café table, the other arm bent at the elbow lifting a Starbucks cup toward her lips."

DEFAULT_ARMWEAR = ""

DEFAULT_POSE_HAND = "One hand resting lightly on the table, the other hand holding a Starbucks cup with fingers elegantly wrapped around it."

DEFAULT_POSE_BODY = "Sitting upright on a café chair, body facing toward the camera, torso frontal, legs together and slightly angled to the side."

DEFAULT_TOP = "Dark navy chiffon dress with thin spaghetti straps, bare shoulders, small cherry blossoms print in soft pink and white, fitted waist, A-line skirt with a deep side slit from mid-thigh exposing the bare leg."

DEFAULT_BOTTOM = ""

DEFAULT_SETTING = "Charming outdoor café terrace, small round table with a Starbucks cup, café chairs, cherry blossom trees lining the street with pink petals falling gently, warm sunny spring day."

DEFAULT_LIGHTING = "Bright even spring daylight, soft frontal natural light, face clearly and brightly lit, no harsh shadows."

DEFAULT_CAMERA = "Full body shot, entire body from head to feet fully in frame, feet and sneakers not cropped, eye-level angle, sharp focus, soft bokeh background."

DEFAULT_POSITIVE_PROMPT = "Ultra HD, 4K, high quality, realistic, detailed, sharp focus, perfect anatomy, beautiful fingers."

DEFAULT_NEGATIVE_PROMPT = "Blurry, low quality, deformed, bad anatomy, extra limbs, ugly, watermark, text, signature, extra fingers, one leg forward, staggered legs, walking pose, weight shift, legs apart, stepping, feet apart, spread legs."

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
    headwear,
    pose_leg,
    legwear,
    pose_foot,
    footwear,
    pose_arm,
    armwear,
    pose_hand,
    pose_body,
    top,
    bottom,
    setting,
    lighting,
    camera,
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [
        subject,
        face,
        pose_head,
        headwear,
        pose_leg,
        legwear,
        pose_foot,
        footwear,
        pose_arm,
        armwear,
        pose_hand,
        pose_body,
        top,
        bottom,
        setting,
        lighting,
        camera,
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


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def load_model(device_name=None):
    """Load and initialize the Qwen-Image model."""
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
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=DTYPE,
    )
    pipe.to(DEVICE)

    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        print("메모리 최적화 적용: model CPU offload, attention slicing (CUDA)")
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
    positive_prompt,
    negative_prompt,
    width,
    height,
    true_cfg_scale,
    num_inference_steps,
    num_images_per_prompt,
    seed,
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

    if not prompt:
        return None, [], [], "오류: 프롬프트를 입력해주세요."

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        # Append positive prompt to main prompt
        if positive_prompt and positive_prompt.strip():
            prompt = prompt.rstrip() + " " + positive_prompt.strip()

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중...")
        print("=" * 60)
        print("[입력 프롬프트]")
        print(prompt)
        print("=" * 60)

        # Print negative prompt
        if negative_prompt and negative_prompt.strip():
            print(f"[네거티브 프롬프트] (true_cfg_scale={true_cfg_scale})")
            print(negative_prompt)
            print("=" * 60)

        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        progress(0.05, desc="추론 시작...")
        print("추론 시작...")

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

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt if negative_prompt and negative_prompt.strip() else None,
            "width": width,
            "height": height,
            "true_cfg_scale": true_cfg_scale,
            "num_inference_steps": steps,
            "num_images_per_prompt": int(num_images_per_prompt),
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
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            gpu_label = gpu_name.replace(" ", "").replace("NVIDIA", "").replace("GeForce", "") + f"-{gpu_mem}GB"
            device_label = gpu_label
        else:
            device_label = DEVICE.upper()
        base_filename = (
            f"{script_name}_{timestamp}_{device_label}_{width}x{height}"
            f"_cfg{true_cfg_scale}_step{steps}_seed{int(seed)}"
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

        saved_info = f"저장됨: {saved_files[0]}" if len(saved_files) == 1 else f"{len(saved_files)}장 저장됨: {saved_files[0]} 외"
        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초")
        for f in saved_files:
            print(f"이미지가 저장되었습니다 : {f}")

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

    with gr.Blocks(
        title="Qwen-Image Text-to-Image Generator",
    ) as interface:
        gr.Markdown("# Qwen-Image Text-to-Image Generator")
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
                        else "모델이 로드되지 않았습니다."
                        " 디바이스를 선택하고 '모델 로드' 버튼을 눌러주세요."
                    ),
                    interactive=False,
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
                prompt_headwear = gr.Textbox(
                    label="4. 머리 장식 (Headwear)",
                    value=DEFAULT_HEADWEAR,
                    lines=1,
                    placeholder="예: black beret, floral hairpin",
                    info="모자, 헤어핀, 머리띠 등 머리 장식을 설명합니다.",
                )
                prompt_pose_leg = gr.Textbox(
                    label="3b. 포즈 - 다리 (Pose: Leg)",
                    value=DEFAULT_POSE_LEG,
                    lines=1,
                    placeholder="예: one leg stepping forward, weight on left leg",
                    info="다리 자세를 설명합니다.",
                )
                prompt_legwear = gr.Textbox(
                    label="8. 레그웨어 (Legwear)",
                    value=DEFAULT_LEGWEAR,
                    lines=1,
                    placeholder="예: thigh-high black stockings, sheer tights",
                    info="스타킹, 양말, 레깅스 등을 설명합니다.",
                )
                prompt_pose_foot = gr.Textbox(
                    label="3c. 포즈 - 발 (Pose: Foot)",
                    value=DEFAULT_POSE_FOOT,
                    lines=1,
                    placeholder="예: feet slightly apart, toes pointed forward",
                    info="발의 위치를 설명합니다.",
                )
                prompt_footwear = gr.Textbox(
                    label="9. 신발 (Footwear)",
                    value=DEFAULT_FOOTWEAR,
                    lines=1,
                    placeholder="예: black stiletto heels, white sneakers",
                    info="신발, 부츠, 샌들 등을 설명합니다.",
                )
                prompt_pose_arm = gr.Textbox(
                    label="3d. 포즈 - 팔 (Pose: Arm)",
                    value=DEFAULT_POSE_ARM,
                    lines=1,
                    placeholder="예: arms resting across torso",
                    info="팔의 위치와 자세를 설명합니다.",
                )
                prompt_armwear = gr.Textbox(
                    label="7. 팔 장식 (Armwear)",
                    value=DEFAULT_ARMWEAR,
                    lines=1,
                    placeholder="예: black lace gloves, silver bracelet",
                    info="장갑, 팔찌, 소매 장식 등을 설명합니다.",
                )
                prompt_pose_hand = gr.Textbox(
                    label="3e. 포즈 - 손 (Pose: Hand)",
                    value=DEFAULT_POSE_HAND,
                    lines=1,
                    placeholder="예: one hand gripping the other arm",
                    info="손의 위치와 동작을 설명합니다.",
                )
                prompt_pose_body = gr.Textbox(
                    label="3f. 포즈 - 몸통 (Pose: Body)",
                    value=DEFAULT_POSE_BODY,
                    lines=2,
                    placeholder="예: body angled slightly, leaning forward",
                    info="몸통 자세와 전체 실루엣을 설명합니다.",
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
                    lines=2,
                    placeholder="예: tiny black panty, mini skirt",
                    info="하의, 속옷 하의 등을 설명합니다.",
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
                            DEFAULT_SUBJECT,
                            DEFAULT_FACE,
                            DEFAULT_POSE_HEAD,
                            DEFAULT_HEADWEAR,
                            DEFAULT_POSE_LEG,
                            DEFAULT_LEGWEAR,
                            DEFAULT_POSE_FOOT,
                            DEFAULT_FOOTWEAR,
                            DEFAULT_POSE_ARM,
                            DEFAULT_ARMWEAR,
                            DEFAULT_POSE_HAND,
                            DEFAULT_POSE_BODY,
                            DEFAULT_TOP,
                            DEFAULT_BOTTOM,
                            DEFAULT_SETTING,
                            DEFAULT_LIGHTING,
                            DEFAULT_CAMERA,
                        ),
                        lines=4,
                        interactive=False,
                        info="위 섹션들이 자동으로 합쳐진 최종 프롬프트입니다.",
                    )
                prompt_sections = [
                    prompt_subject,
                    prompt_face,
                    prompt_pose_head,
                    prompt_headwear,
                    prompt_pose_leg,
                    prompt_legwear,
                    prompt_pose_foot,
                    prompt_footwear,
                    prompt_pose_arm,
                    prompt_armwear,
                    prompt_pose_hand,
                    prompt_pose_body,
                    prompt_top,
                    prompt_bottom,
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

                positive_prompt_box = gr.Textbox(
                    label="포지티브 프롬프트 (Positive Prompt)",
                    value=DEFAULT_POSITIVE_PROMPT,
                    lines=2,
                    placeholder="예: masterpiece, best quality, highly detailed",
                    info="최종 프롬프트 뒤에 추가로 덧붙일 키워드를 입력합니다.",
                )
                negative_prompt_box = gr.Textbox(
                    label="네거티브 프롬프트 (Negative Prompt)",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=2,
                    placeholder="예: blurry, deformed hands, bad anatomy",
                    info="생성에서 제외할 요소를 입력합니다.",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 파라미터 설정")
                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=928,
                        info="이미지 너비 (픽셀). 64의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1664,
                        info="이미지 높이 (픽셀). 64의 배수.",
                    )

                with gr.Row():
                    true_cfg_scale = gr.Slider(
                        label="True CFG Scale (프롬프트/네거티브 강도)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=4.0,
                        info="프롬프트 준수도 및 네거티브 프롬프트 강도. Qwen 권장: 4.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=100,
                        step=1,
                        value=50,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 50",
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
                    output_gallery = gr.Gallery(label="개별 이미지", columns=[1, 1, 2, 2], rows=[1, 1, 1, 2], object_fit="contain", allow_preview=True)
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
                combined_prompt,
                positive_prompt_box,
                negative_prompt_box,
                width,
                height,
                true_cfg_scale,
                num_inference_steps,
                num_images_per_prompt,
                seed,
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
