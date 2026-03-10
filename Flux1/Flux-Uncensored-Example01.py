import re
import torch
import platform
from diffusers import FluxPipeline
from datetime import datetime
import os
import gc
import signal
import sys
import psutil
import time
import gradio as gr

# Default values for each prompt section
DEFAULT_SUBJECT = "A photorealistic, high-quality full body portrait of a young Korean woman with a soft, idol aesthetic."

DEFAULT_APPEARANCE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera. She has long, wavy jet-black hair with thick, straight-cut bangs (fringe) that frame her face."

DEFAULT_POSE = "Standing in a sexy, confident pose at the shoreline. One hand gently running through her hair, the other resting on her hip. Her body slightly arched, showing off her curves. Feet in the shallow water with waves lapping around her ankles."

DEFAULT_OUTFIT = "fully naked, no clothing, no accessories, no jewelry, natural beauty. Emphasize her flawless skin and natural form without any coverings."

DEFAULT_SETTING = "Tropical beach at golden hour. Pristine white sand, crystal-clear turquoise ocean water, gentle waves, lush palm trees softly blurred in the background."

DEFAULT_LIGHTING = "Warm golden hour sunlight casting a soft glow on her skin. Glowing rim light from the setting sun behind her, creating a beautiful halo effect. Warm, sensual, cinematic atmosphere."

DEFAULT_CAMERA = "85mm portrait lens, full body shot, tack-sharp focus on subject, shallow depth of field with soft bokeh background. Photorealistic, gravure photography style, 8k resolution, masterpiece. Perfect anatomy, High-fidelity skin textures."

DEFAULT_NEGATIVE_PROMPT = "Extra hands, extra legs, extra feet, extra arms, waist pleats, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low resolution, normal quality,((monochrome)), ((grayscale)), skin spots, wet, acnes, skin blemishes, age spot, man boobs, backlight, mutated hands, (poorly drawn hands:1.33), blurry, (bad anatomy:1.21), (bad proportions:1.33), extra limbs, (disfigured:1.33), (more than 2 nipples:1.33), (missing arms:1.33), (extra legs:1.33), (fused fingers:1.61), (too many fingers:1.61), (unclear eyes:1.33), lowers, bad hands, missing fingers, extra digit, (futa:1.1), bad hands, missing fingers, (cleft chin:1.3)"


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    subject, appearance, pose, outfit, setting, lighting, camera
):
    """Combine separate prompt sections into one final prompt string."""
    sections = [subject, appearance, pose, outfit, setting, lighting, camera]
    combined = ", ".join(normalize_spacing(s) for s in sections if s and s.strip())
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
    """Load and initialize the FLUX.1-dev model with uncensored LoRA (T5-XXL only, CLIP disabled)."""
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
    print("T5-XXL 텍스트 인코더만 사용합니다. (CLIP 비활성화)")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
    )
    pipe.load_lora_weights(
        "enhanceaiteam/Flux-uncensored-v2", weight_name="lora.safetensors", prefix=None
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
        pipe.enable_attention_slicing()
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)
        elif hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
        print("메모리 최적화 적용: attention slicing (MPS)")

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    true_cfg_scale,
    num_inference_steps,
    seed,
    max_sequence_length,
    image_format,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate image from text prompt using T5-XXL encoding only."""
    global pipe

    if pipe is None:
        return (
            None,
            "오류: 모델이 로드되지 않았습니다. '모델 로드' 버튼을 먼저 눌러주세요.",
        )

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="프롬프트 인코딩 중...")
        print("프롬프트 인코딩 중 (T5-XXL)...")

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # Encode prompt using T5-XXL only
        max_len = int(max_sequence_length)
        text_inputs = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )

        # Count tokens to detect clipping
        raw_ids = pipe.tokenizer_2(prompt, truncation=False, return_tensors="pt")[
            "input_ids"
        ][0]
        raw_token_count = len(raw_ids)
        clipped = max(0, raw_token_count - max_len)
        if clipped > 0:
            print(f"T5 토큰 수: {raw_token_count} / {max_len} → {clipped}개 잘림!")
            truncated_text = pipe.tokenizer_2.decode(
                raw_ids[max_len:], skip_special_tokens=True
            )
            print(f"[잘린 텍스트]: {truncated_text}")
        else:
            print(f"T5 토큰 수: {raw_token_count} / {max_len} (잘림 없음)")

        with torch.inference_mode():
            prompt_embeds = pipe.text_encoder_2(
                text_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=DTYPE)

        # Zero pooled embeddings (CLIP disabled)
        pooled_prompt_embeds = torch.zeros(
            1, 768, dtype=DTYPE, device=prompt_embeds.device
        )

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

        # Build pipeline kwargs with pre-computed T5 embeddings
        pipe_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": float(true_cfg_scale),
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        # Add negative prompt embeddings if provided and true_cfg_scale > 1
        if negative_prompt and float(true_cfg_scale) > 1.0:
            neg_inputs = pipe.tokenizer_2(
                negative_prompt,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            with torch.inference_mode():
                neg_embeds = pipe.text_encoder_2(
                    neg_inputs["input_ids"].to(DEVICE),
                    output_hidden_states=False,
                )[0]
            pipe_kwargs["negative_prompt_embeds"] = neg_embeds.to(dtype=DTYPE)
            pipe_kwargs["negative_pooled_prompt_embeds"] = torch.zeros(
                1, 768, dtype=DTYPE, device=prompt_embeds.device
            )

        # Run the pipeline
        with torch.inference_mode():
            image = pipe(**pipe_kwargs).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        ext = "jpg" if image_format == "JPEG" else "png"
        filename = (
            f"{script_name}_{timestamp}_{DEVICE.upper()}_{width}x{height}"
            f"_gs{guidance_scale}_tcfg{true_cfg_scale}_step{steps}_seed{int(seed)}.{ext}"
        )

        token_info = (
            f"토큰: {raw_token_count}/{max_len} → {clipped}개 잘림!"
            if clipped > 0
            else f"토큰: {raw_token_count}/{max_len}"
        )
        print(f"이미지 생성 완료! 소요 시간: {elapsed:.1f}초 | {token_info}")
        print(f"이미지가 저장되었습니다 : {filename}")
        if image_format == "JPEG":
            image.save(filename, format="JPEG", quality=100, subsampling=0)
        else:
            image.save(filename)

        progress(1.0, desc="완료!")
        return (
            image,
            f"✓ 완료! ({elapsed:.1f}초) | {token_info} | 저장됨: {filename}",
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
        title="Flux.1-dev Uncensored Text-to-Image Generator",
    ) as interface:
        gr.Markdown("# Flux.1-dev Uncensored Text-to-Image Generator")
        gr.Markdown(
            f"Uncensored LoRA를 사용하여 텍스트에서 이미지를 생성합니다."
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
                    value=DEFAULT_SUBJECT,
                    lines=2,
                    placeholder="예: A beautiful young woman, full body portrait",
                    info="이미지의 주된 주제나 대상을 설명합니다.",
                )
                prompt_appearance = gr.Textbox(
                    label="2. 외모 (Appearance)",
                    value=DEFAULT_APPEARANCE,
                    lines=2,
                    placeholder="예: Fair skin, long dark hair, innocent expression",
                    info="인물의 외모, 얼굴, 머리카락, 나이 등을 설명합니다.",
                )
                prompt_pose = gr.Textbox(
                    label="3. 포즈/구도 (Pose & Composition)",
                    value=DEFAULT_POSE,
                    lines=2,
                    placeholder="예: Standing naturally, looking at the camera",
                    info="자세, 시선 방향, 카메라 앵글, 촬영 구도를 설명합니다.",
                )
                prompt_outfit = gr.Textbox(
                    label="4. 의상 (Outfit)",
                    value=DEFAULT_OUTFIT,
                    lines=2,
                    placeholder="예: Wearing a white dress",
                    info="의상, 액세서리, 착용한 아이템을 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="5. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    placeholder="예: White room, sheer curtains, city view",
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="6. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    placeholder="예: Natural daylight, high-key, soft diffused light",
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="7. 카메라/스타일 (Camera & Style)",
                    value=DEFAULT_CAMERA,
                    lines=2,
                    placeholder="예: 85mm, photorealistic, boudoir, 8k, masterpiece",
                    info="카메라 설정, 렌즈, 화질, 스타일 키워드를 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            DEFAULT_SUBJECT,
                            DEFAULT_APPEARANCE,
                            DEFAULT_POSE,
                            DEFAULT_OUTFIT,
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
                    prompt_appearance,
                    prompt_pose,
                    prompt_outfit,
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

                negative_prompt_input = gr.Textbox(
                    label="부정 프롬프트 (Negative Prompt)",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                    info="이미지에 포함되지 않기를 원하는 요소 (True CFG Scale > 1 필요)",
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
                        info="이미지 너비 (픽셀). 64의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1536,
                        info="이미지 높이 (픽셀). 64의 배수.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=3.5,
                        info="프롬프트 준수도. 낮으면 창의적, 높으면 정확. 권장: 3.5-7",
                    )
                    true_cfg_scale = gr.Slider(
                        label="True CFG Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.1,
                        value=1.0,
                        info=">1 에서 부정 프롬프트와 함께 CFG 활성화. 권장: 1.0-3.5",
                    )

                with gr.Row():
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28,
                        info="생성 단계 수. 높으면 품질 향상, 시간 증가. 권장: 20-28",
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
                        step=64,
                        value=512,
                        info="텍스트 인코더 최대 길이. 긴 프롬프트는 높은 값 필요.",
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
                output_image = gr.Image(label="생성된 이미지", height=700)
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
                combined_prompt,
                negative_prompt_input,
                width,
                height,
                guidance_scale,
                true_cfg_scale,
                num_inference_steps,
                seed,
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
