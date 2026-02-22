import re
import torch
import os
import gc
import sys
import signal
import time
import platform
import psutil
from datetime import datetime
from diffusers import ZImagePipeline
import gradio as gr

# https://prompthero.com/prompt/a885d64532b-flux-flux-11-pro-ultra-a-professional-photo-of-a-exceptionally-of-the-most-beautiful-russian-girl-in-the-world-age-18-futuristicmasterpiece-realistic-perfect

# Default values for each prompt section
DEFAULT_QUALITY = "a professional photo, (futuristic), (masterpiece), (realistic), (perfect anatomy), 8k, highly detailed, full length frame, High detail RAW color art, sharp focus, hyper realism."
DEFAULT_NEGATIVE = "extra hands, extra legs, extra feet, extra arms, Waist Pleats, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, wet, acnes, skin blemishes, age spot, manboobs, backlight, mutated hands, (poorly drawn hands:1.33), blurry, (bad anatomy:1.21), (bad proportions:1.33), extra limbs, (disfigured:1.33), (more than 2 nipples:1.33), (missing arms:1.33), (extra legs:1.33), (fused fingers:1.61), (too many fingers:1.61), (unclear eyes:1.33), lowers, bad hands, missing fingers, extra digit, (futa:1.1), bad hands, missing fingers, (cleft chin:1.3), exposed nipples"
DEFAULT_APPEARANCE = "exceptionally beautiful Russian girl, age 18, very long strawberry blonde hair, stunning blue eyes, black eye liner."
DEFAULT_OUTFIT = "wearing a very tiny string bikini top, and tiny G-string bikini bottoms."
DEFAULT_POSE = "standing on the beach, full body visible, front-facing the camera."
DEFAULT_SETTING = "sunny beach, pristine white sandy shore, sparkling turquoise ocean in the background, clear blue sky."
DEFAULT_LIGHTING = "diffused soft lighting, shallow depth of field, cinematic lighting."
DEFAULT_CAMERA = "Canon EOS R5, 35mm f/5.6, ISO 200, 1/500s shutter. Full body visible with beach background clearly in focus."


def normalize_spacing(text: str) -> str:
    """Normalize whitespace around punctuation in a prompt string."""
    text = re.sub(r"([,.:;])(?!\s)", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def combine_prompt_sections(
    quality, appearance, outfit, pose, setting, lighting, camera
):
    """Combine positive prompt sections into one final prompt string."""
    sections = [quality, appearance, outfit, pose, setting, lighting, camera]
    combined = ", ".join(normalize_spacing(s) for s in sections if s and s.strip())
    return combined

# Global variables
pipe = None
interface = None


def detect_device():
    """Auto-detect the best available device and data type."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32


DEVICE, DTYPE = detect_device()
IS_MPS = DEVICE == "mps"


def print_hardware_info():
    """Print detailed hardware specifications."""
    print("=" * 60)
    print("하드웨어 사양")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
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

    print("-" * 60)
    print("GPU 정보")
    print("-" * 60)
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  - VRAM: {props.total_memory / (1024**3):.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) 사용 가능")
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
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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


def load_model(selected_device=None):
    """Load and initialize the Z-Image model with optimizations."""
    global pipe, DEVICE, DTYPE

    if selected_device is not None and not IS_MPS:
        DEVICE = selected_device
        DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

    if pipe is not None:
        print("기존 모델 해제 중...")
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"모델 로딩 중... (Device: {DEVICE}, dtype: {DTYPE})")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    pipe.to(DEVICE)

    if DEVICE == "cuda":
        print("CUDA 최적화 활성화됨")
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    elif DEVICE == "mps":
        print("MPS 최적화 활성화 안함")
    else:
        print("CPU 모드로 실행 중. 최적화는 제한적입니다.")
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"


def generate_image(
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    cfg_normalization,
    seed,
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
        steps = int(num_inference_steps)
        start_time = time.time()

        progress(0.0, desc="이미지 생성 준비 중...")

        gen_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(int(seed))

        progress(0.05, desc="추론 시작...")

        # Callback to report each inference step to Gradio progress bar
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.85
            progress(
                progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)"
            )
            return callback_kwargs

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
        ).images[0]

        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp and parameters
        elapsed = time.time() - start_time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}_{DEVICE}_{int(width)}x{int(height)}_gs{guidance_scale}_step{steps}_cfgnorm{cfg_normalization}_seed{int(seed)}.png"

        image.save(filename)
        print(f"이미지가 저장되었습니다: {filename}")

        progress(1.0, desc="완료!")
        return image, f"✓ 완료! ({elapsed:.1f}초) 저장됨: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


def main():
    global interface

    print_hardware_info()

    print(f"\n자동으로 감지된 디바이스: {DEVICE} (dtype: {DTYPE})")
    load_model()

    with gr.Blocks(title="Z-Image Text-to-Image Generator") as interface:
        gr.Markdown("# Z-Image Text-to-Image Generator")
        gr.Markdown(
            f"Tongyi-MAI/Z-Image 모델을 사용하여 텍스트에서 이미지를 생성합니다. (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                if not IS_MPS:
                    device_choices = []
                    if torch.cuda.is_available():
                        device_choices.append("cuda")
                    device_choices.append("cpu")
                    device_selector = gr.Dropdown(
                        label="디바이스 선택",
                        choices=device_choices,
                        value=DEVICE,
                        info="모델을 로드할 디바이스를 선택합니다. 변경 후 '모델 로드' 버튼을 눌러주세요.",
                    )
                else:
                    device_selector = gr.Textbox(
                        label="디바이스",
                        value="MPS (Apple Silicon)",
                        interactive=False,
                    )
                load_model_btn = gr.Button("모델 로드", variant="secondary")
                model_status = gr.Textbox(
                    label="모델 상태",
                    value=(
                        f"모델 로딩 완료! (Device: {DEVICE}, dtype: {DTYPE})"
                        if pipe is not None
                        else "모델이 로드되지 않았습니다."
                    ),
                    interactive=False,
                )

                gr.Markdown("### 프롬프트 구성")
                prompt_quality = gr.Textbox(
                    label="1. 품질/해상도 (Quality & Resolution)",
                    value=DEFAULT_QUALITY,
                    lines=2,
                    info="이미지의 품질, 해상도, 스타일 관련 키워드입니다.",
                )
                prompt_negative = gr.Textbox(
                    label="2. 네거티브 프롬프트 (Negative Prompt)",
                    value=DEFAULT_NEGATIVE,
                    lines=2,
                    info="생성에서 제외할 요소들입니다. 파이프라인에 별도로 전달됩니다.",
                )
                prompt_appearance = gr.Textbox(
                    label="3. 외모 (Appearance)",
                    value=DEFAULT_APPEARANCE,
                    lines=2,
                    info="인물의 외모, 얼굴, 머리카락, 나이 등을 설명합니다.",
                )
                prompt_outfit = gr.Textbox(
                    label="4. 의상 (Outfit)",
                    value=DEFAULT_OUTFIT,
                    lines=2,
                    info="의상, 액세서리, 착용한 아이템을 설명합니다.",
                )
                prompt_pose = gr.Textbox(
                    label="5. 포즈/구도 (Pose & Composition)",
                    value=DEFAULT_POSE,
                    lines=2,
                    info="자세, 시선 방향, 카메라 앵글을 설명합니다.",
                )
                prompt_setting = gr.Textbox(
                    label="6. 배경/장소 (Setting & Background)",
                    value=DEFAULT_SETTING,
                    lines=2,
                    info="배경, 장소, 환경, 계절 등을 설명합니다.",
                )
                prompt_lighting = gr.Textbox(
                    label="7. 조명 (Lighting)",
                    value=DEFAULT_LIGHTING,
                    lines=2,
                    info="조명 조건, 빛의 방향, 분위기를 설명합니다.",
                )
                prompt_camera = gr.Textbox(
                    label="8. 카메라 설정 (Camera Settings)",
                    value=DEFAULT_CAMERA,
                    lines=2,
                    info="카메라 기종, 렌즈, ISO, 셔터 스피드 등을 설명합니다.",
                )
                with gr.Accordion("최종 프롬프트 (Combined Prompt)", open=False):
                    combined_prompt = gr.Textbox(
                        label="최종 프롬프트",
                        value=combine_prompt_sections(
                            DEFAULT_QUALITY,
                            DEFAULT_APPEARANCE,
                            DEFAULT_OUTFIT,
                            DEFAULT_POSE,
                            DEFAULT_SETTING,
                            DEFAULT_LIGHTING,
                            DEFAULT_CAMERA,
                        ),
                        lines=4,
                        interactive=False,
                        info="위 섹션들이 자동으로 합쳐진 최종 프롬프트입니다.",
                    )
                prompt_sections = [
                    prompt_quality,
                    prompt_appearance,
                    prompt_outfit,
                    prompt_pose,
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

                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=768,
                        info="생성할 이미지의 너비 (픽셀). 64의 배수.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1536,
                        info="생성할 이미지의 높이 (픽셀). 64의 배수.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.1,
                        value=4.0,
                        info="프롬프트 충실도. 낮을수록 창의적, 높을수록 정확. 권장: 4.0",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25,
                        info="이미지 생성 단계 수. 높을수록 품질 향상. 권장: 25",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        minimum=0,
                        maximum=1000,
                        value=42,
                        precision=0,
                        info="같은 시드 = 같은 결과. 재현성을 위해 사용합니다.",
                    )
                    cfg_normalization = gr.Checkbox(
                        label="CFG Normalization",
                        value=False,
                        info="CFG 정규화 사용 여부.",
                    )

                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        load_model_btn.click(
            fn=load_model,
            inputs=[device_selector],
            outputs=[model_status],
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                combined_prompt,
                prompt_negative,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                cfg_normalization,
                seed,
            ],
            outputs=[output_image, output_message],
        )

    interface.launch(inbrowser=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
