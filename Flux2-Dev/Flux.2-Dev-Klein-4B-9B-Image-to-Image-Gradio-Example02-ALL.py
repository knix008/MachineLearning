import torch
import platform
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import signal
import sys
import inspect
import gradio as gr

DEFAULT_PROMPT = "Turn her body to the front side. Change her hair to black color."

DEFAULT_IMAGE = "sample02.png"

DEFAULT_MODEL = "FLUX.2-klein-4B"
MODEL_OPTIONS = {
    "FLUX.2-klein-4B": "black-forest-labs/FLUX.2-klein-4B",
    "FLUX.2-klein-9B": "black-forest-labs/FLUX.2-klein-9B",
}
current_model_name = None


def get_device_and_dtype():
    """Detect the best available device and appropriate data type."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"Using CUDA (GPU): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16  # MPS has limited bfloat16 support
        print(f"Using MPS (Apple Silicon): {platform.processor()}")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"Using CPU: {platform.processor()}")

    print(f"Data type: {dtype}")
    return device, dtype


# Global variables for model
DEVICE, DTYPE = get_device_and_dtype()
pipe = None
demo = None


def cleanup():
    """Clean up resources before exit."""
    global pipe, demo
    print("\n프로그램 종료 중...")

    if demo is not None:
        try:
            demo.close()
            print("Gradio 서버 종료됨")
        except:
            pass

    if pipe is not None:
        del pipe
        print("모델 메모리 해제됨")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA 캐시 정리됨")

    print("종료 완료!")


def signal_handler(sig, frame):
    """Handle Ctrl+C signal."""
    cleanup()
    sys.exit(0)

def load_model(model_name=DEFAULT_MODEL):
    """Load and initialize the Flux2 Img2Img model with optimizations."""
    global pipe, current_model_name
    import gc

    # Skip if same model is already loaded
    if current_model_name == model_name and pipe is not None:
        print(f"모델 '{model_name}'이 이미 로드되어 있습니다.")
        return pipe

    # Release previous model if loaded
    if pipe is not None:
        print(f"기존 모델 '{current_model_name}' 해제 중...")
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    model_path = MODEL_OPTIONS[model_name]
    print(f"모델 로딩 중: {model_name} ({model_path})...")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=DTYPE
    )
    pipe = pipe.to(DEVICE)

    if DEVICE == "cuda" or DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
    elif DEVICE == "mps":
        pass
    else:
        print("No valid device found!!!")
        exit(1)

    current_model_name = model_name
    print(f"모델 로딩 완료: {model_name} (Device: {DEVICE})")

    # Print all supported arguments
    print("\n" + "="*60)
    print("파이프라인 지원 인자 목록:")
    print("="*60)
    sig = inspect.signature(pipe.__call__)
    for param_name, param in sig.parameters.items():
        default = param.default
        if default is inspect.Parameter.empty:
            default_str = "(필수)"
        elif default is None:
            default_str = "= None"
        else:
            default_str = f"= {default}"
        print(f"  - {param_name}: {default_str}")
    print("="*60 + "\n")

    return pipe


def switch_model(model_name):
    """Switch to a different model and return status."""
    load_model(model_name)
    return f"모델 변경 완료: {model_name}"

def generate_image(input_image, prompt, negative_prompt, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image from input image and text prompt."""
    global pipe

    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        # Convert to PIL Image if needed
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)

        # Setup generator (MPS doesn't support Generator directly, use CPU)
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        print(f"출력 크기: {width}x{height}")
        print(f"추론 스텝: {num_inference_steps}, 시드: {int(seed)}")
        print(f"이미지 편집 중... (steps: {num_inference_steps}, seed: {int(seed)})")

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "image": input_image,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        image = pipe(**pipe_kwargs).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{current_model_name}_{timestamp}_step{num_inference_steps}_seed{int(seed)}.png"
        image.save(output_path)
        print(f"이미지 저장됨: {output_path}")

        return image, f"✓ 이미지 편집 완료! 저장됨: {output_path}"

    except Exception as e:
        return None, f"오류: {str(e)}"

def main():
    global demo

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Load model once at startup
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.2 Dev Klein 4B/9B Image-to-Image 편집기") as demo:
        gr.Markdown("# Flux.2 Dev Klein 4B/9B Image-to-Image 편집기")
        gr.Markdown(f"이미지를 업로드하고 프롬프트로 편집하세요. (Device: **{DEVICE.upper()}**)")

        with gr.Row():
            with gr.Column():
                model_selector = gr.Radio(
                    label="모델 선택",
                    choices=list(MODEL_OPTIONS.keys()),
                    value=DEFAULT_MODEL,
                    info="4B: 빠르고 가벼움 / 9B: 고품질, 더 많은 메모리 필요",
                )
                load_model_btn = gr.Button("모델 로드", variant="secondary")
                model_status = gr.Textbox(
                    label="모델 상태",
                    value=f"현재 모델: {DEFAULT_MODEL}",
                    interactive=False,
                )
                input_image = gr.Image(
                    label="입력 이미지",
                    type="pil",
                    value=DEFAULT_IMAGE,
                    height=800
                )
                prompt_input = gr.Textbox(
                    label="프롬프트 (영어)",
                    info="편집할 내용을 설명하세요",
                    placeholder="예: change the background to a beach at sunset",
                    value=DEFAULT_PROMPT,
                    lines=3
                )
                negative_prompt_input = gr.Textbox(
                    label="네거티브 프롬프트 (영어)",
                    info="생성하지 않을 내용을 설명하세요",
                    placeholder="예: blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text",
                    value="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text",
                    lines=2
                )

                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        height_input = gr.Slider(
                            label="높이 (Height)",
                            info="생성할 이미지의 높이 (픽셀)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024
                        )
                        width_input = gr.Slider(
                            label="너비 (Width)",
                            info="생성할 이미지의 너비 (픽셀)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=768
                        )
                    
                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=1.0
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=4
                        )

                    seed_input = gr.Slider(
                        label="시드 (Seed)",
                        info="재현성을 위한 난수 시드 (-1: 랜덤)",
                        minimum=-1,
                        maximum=1000,
                        step=1,
                        value=42
                    )

                submit_btn = gr.Button("이미지 편집", variant="primary", size="lg")

            with gr.Column():
                image_output = gr.Image(label="출력 이미지", height=1200)
                status_output = gr.Textbox(label="상태", interactive=False)

        # Load model when button is clicked
        load_model_btn.click(
            fn=switch_model,
            inputs=[model_selector],
            outputs=[model_status],
        )

        # Connect button to generation function
        submit_btn.click(
            fn=generate_image,
            inputs=[
                input_image,
                prompt_input,
                negative_prompt_input,
                height_input,
                width_input,
                guidance_input,
                steps_input,
                seed_input
            ],
            outputs=[image_output, status_output]
        )

    # Launch the interface
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
