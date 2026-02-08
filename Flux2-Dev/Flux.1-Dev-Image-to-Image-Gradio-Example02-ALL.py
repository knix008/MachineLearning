import torch
import platform
from diffusers import FluxImg2ImgPipeline
from datetime import datetime
from PIL import Image
import os
import gc
import signal
import sys
import psutil
import gradio as gr
import inspect

DEFAULT_PROMPT = "Change her body to the the viewer. Change her hair to black color. Perfect anatomy and proportions, detailed face, intricate details, high quality, 4k."

DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, deformed hands, extra fingers, bad anatomy, disfigured, poorly drawn face, mutation, mutated, ugly, watermark, text, signature"


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
supports_negative_prompt = False


def print_hardware_info():
    """Print detailed hardware specifications."""
    print("=" * 60)
    print("하드웨어 사양")
    print("=" * 60)

    # OS 정보
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OS 버전: {platform.version()}")
    print(f"아키텍처: {platform.machine()}")

    # Python 정보
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # CPU 정보
    print("-" * 60)
    print("CPU 정보")
    print("-" * 60)
    print(f"프로세서: {platform.processor()}")
    print(f"물리 코어: {psutil.cpu_count(logical=False)}")
    print(f"논리 코어: {psutil.cpu_count(logical=True)}")

    # 메모리 정보
    mem = psutil.virtual_memory()
    print("-" * 60)
    print("메모리 정보")
    print("-" * 60)
    print(f"총 RAM: {mem.total / (1024**3):.1f} GB")
    print(f"사용 가능: {mem.available / (1024**3):.1f} GB")
    print(f"사용률: {mem.percent}%")

    # GPU 정보
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


def load_model():
    """Load and initialize the Flux model with optimizations."""
    global pipe, supports_negative_prompt

    print("모델 로딩 중...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE, low_cpu_mem_usage=True
    )

    # Enable memory optimizations based on device
    if DEVICE == "cuda" or DEVICE == "cpu":
        pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    elif DEVICE == "mps":
        # MPS doesn't support cpu_offload well
        pass
    else:
        print("No valid device found!!!")
        exit(1)

    # Check negative prompt support
    pipe_params = inspect.signature(pipe.__call__).parameters
    supports_negative_prompt = "negative_prompt" in pipe_params
    if supports_negative_prompt:
        print("네거티브 프롬프트: 지원됨 ✓")
    else:
        print(
            "네거티브 프롬프트: 지원되지 않음 (이 파이프라인은 negative_prompt 파라미터를 지원하지 않습니다. 입력값이 무시됩니다.)"
        )

    print(f"모델 로딩 완료! (Device: {DEVICE})")
    return pipe


def generate_image(
    input_image,
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    strength,
    max_sequence_length,
):
    global pipe

    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)

        input_image = input_image.convert("RGB")

        if width and height:
            input_image = input_image.resize((int(width), int(height)))

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(int(seed))

        # Run the pipeline
        pipe_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "max_sequence_length": int(max_sequence_length),
        }
        neg_prompt_msg = ""
        if negative_prompt:
            if supports_negative_prompt:
                pipe_kwargs["negative_prompt"] = negative_prompt
            else:
                neg_prompt_msg = " (⚠ 네거티브 프롬프트는 이 파이프라인에서 지원되지 않아 무시되었습니다)"
                print(
                    "경고: 네거티브 프롬프트가 무시됨 - 이 파이프라인은 negative_prompt를 지원하지 않습니다."
                )
        print("Generating image with the prompt :", pipe_kwargs["prompt"])
        image = pipe(**pipe_kwargs).images[0]

        # Save with timestamp and parameters
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        params = f"w{int(width)}_h{int(height)}_gs{guidance_scale}_steps{int(num_inference_steps)}_seed{int(seed)}_str{strength}_msl{int(max_sequence_length)}"
        filename = f"{script_name}_{timestamp}_{params}.png"
        image.save(filename)

        print(f"이미지 저장됨: {filename}")
        return image, f"✓ 이미지가 저장되었습니다: {filename}{neg_prompt_msg}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


def update_size_from_image(image):
    """Update width and height sliders based on input image dimensions."""
    if image is None:
        return gr.update(), gr.update()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    w, h = image.size
    # Round to nearest 64 and clamp to slider range [256, 2048]
    w = max(256, min(2048, round(w / 64) * 64))
    h = max(256, min(2048, round(h / 64) * 64))
    return gr.update(value=w), gr.update(value=h)


def main():
    global interface

    # Print hardware specifications
    print_hardware_info()

    # Load model once at startup
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.1 Dev Image Generator") as interface:
        gr.Markdown("# Flux.1 Dev Image-to-Image Editor")
        gr.Markdown(
            f"AI를 사용하여 입력 이미지를 프롬프트로 편집합니다. (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="입력 이미지", type="pil", height=400, value="test01.png"
                )

                # Input parameters
                prompt = gr.Textbox(
                    label="프롬프트",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    placeholder="이미지에 대한 설명을 입력하세요 (77단어 이하 권장)",
                    info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다. 예: '여자, 미소, 해변, 빨간 비키니'",
                )

                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=2,
                    placeholder="원하지 않는 요소를 입력하세요",
                    info="생성된 이미지에서 제외하고 싶은 요소를 설명합니다. 예: 'low quality, blurry, deformed hands, extra fingers'",
                )

                with gr.Row():
                    width = gr.Slider(
                        label="이미지 너비",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=512,
                        info="생성할 이미지의 너비를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                    )
                    height = gr.Slider(
                        label="이미지 높이",
                        minimum=256,
                        maximum=2048,
                        step=64,
                        value=1024,
                        info="생성할 이미지의 높이를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                    )

                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale (프롬프트 강도)",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=3.5,
                        info="모델이 프롬프트를 얼마나 따를지 제어합니다. 낮을수록 창의적, 높을수록 정확합니다. 권장: 4-15",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=10,
                        maximum=50,
                        step=1,
                        value=28,
                        info="이미지 생성 과정의 단계 수입니다. 높을수록 품질이 좋지만 시간이 더 걸립니다. 권장: 20-28",
                    )

                with gr.Row():
                    seed = gr.Number(
                        label="시드",
                        value=42,
                        precision=0,
                        info="난수 생성의 시작점입니다. 같은 시드를 사용하면 같은 결과를 얻습니다.",
                    )
                    strength = gr.Slider(
                        label="강도",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.1,
                        value=0.5,
                        info="진정한 편집 효과를 원한다면 0.3~0.7 정도로 낮추는 것이 좋습니다. strength가 낮을수록 원본 이미지의 구조를 더 많이 유지합니다.",
                    )

                with gr.Row():
                    max_sequence_length = gr.Slider(
                        label="Max Sequence Length",
                        minimum=128,
                        maximum=512,
                        step=128,
                        value=512,
                        info="텍스트 인코더의 최대 토큰 시퀀스 길이입니다. 낮을수록 메모리를 절약하지만 긴 프롬프트가 잘릴 수 있습니다.",
                    )

                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        # Update width/height sliders when input image changes
        input_image.change(
            fn=update_size_from_image,
            inputs=[input_image],
            outputs=[width, height],
        )

        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_image,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                width,
                height,
                guidance_scale,
                num_inference_steps,
                seed,
                strength,
                max_sequence_length,
            ],
            outputs=[output_image, output_message],
        )

    # Launch the interface
    interface.launch(inbrowser=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
