import warnings
import logging
import os
import gc
import signal
import sys
import inspect
import time
import platform
from datetime import datetime
from diffusers import FluxImg2ImgPipeline
from PIL import Image
import torch
import gradio as gr

warnings.filterwarnings("ignore", message=".*No LoRA keys associated.*")
logging.getLogger("diffusers").setLevel(logging.ERROR)

DEFAULT_PROMPT = "Make her naked."

DEFAULT_NEGATIVE_PROMPT = "Extra hands, extra legs, extra feet, extra arms, extra toes, Waist Pleats, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low resolution, normal quality, ((monochrome)), ((grayscale)), skin spots, wet, acnes, skin blemishes, age spot, man boobs, backlight, mutated hands, (poorly drawn hands:1.33), blurry, (bad anatomy:1.21), (bad proportions:1.33), extra limbs, (disfigured:1.33), (more than 2 nipples:1.33), (missing arms:1.33), (extra legs:1.33), (fused fingers:1.61), (too many fingers:1.61), (unclear eyes:1.33), lowers, bad hands, missing fingers, extra digit, (futa:1.1), bad hands, missing fingers, (cleft chin:1.3)"


def get_device_and_dtype():
    """Detect the best available device and appropriate data type."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"Using CUDA (GPU): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16
        print(f"Using MPS (Apple Silicon): {platform.processor()}")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"Using CPU: {platform.processor()}")

    print(f"Data type: {dtype}")
    return device, dtype


# Global variables
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
        except Exception:
            pass

    if pipe is not None:
        del pipe
        print("모델 메모리 해제됨")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA 캐시 정리됨")

    gc.collect()
    print("종료 완료!")


def signal_handler(sig, frame):
    """Handle Ctrl+C signal."""
    cleanup()
    sys.exit(0)


def load_model():
    """Load and initialize the FLUX.1-dev img2img model with uncensored LoRA."""
    global pipe

    print("모델 로딩 중...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=DTYPE,
    )
    pipe.load_lora_weights(
        "enhanceaiteam/Flux-uncensored-v2", weight_name="lora.safetensors", prefix=None
    )
    pipe.to(DEVICE)

    # Memory optimization based on device
    if DEVICE == "cuda" or DEVICE == "cpu":
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(
            "메모리 최적화 적용: sequential CPU offload, model CPU offload, attention slicing"
        )
    elif DEVICE == "mps":
        print("MPS 디바이스: 추가 메모리 최적화 없음")
    else:
        print(
            f"알 수 없는 디바이스 유형: {DEVICE}. 최적화가 적용되지 않을 수 있습니다."
        )
        exit(1)

    print(f"모델 로딩 완료! (Device: {DEVICE})")

    # Print all supported arguments
    print("\n" + "=" * 60)
    print("파이프라인 지원 인자 목록:")
    print("=" * 60)
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
    print("=" * 60 + "\n")

    return pipe


def on_image_upload(image):
    """Update width/height sliders when an image is uploaded."""
    if image is None:
        return gr.update(), gr.update()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    w, h = image.size
    # Round to nearest 64
    w = (w // 64) * 64
    h = (h // 64) * 64
    w = max(256, min(2048, w))
    h = max(256, min(2048, h))
    return gr.update(value=w), gr.update(value=h)


def generate_image(
    input_image,
    prompt,
    negative_prompt,
    strength,
    width,
    height,
    num_inference_steps,
    guidance_scale,
    true_cfg_scale,
    max_sequence_length,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate image from input image + text prompt."""
    global pipe

    if input_image is None:
        return None, "오류: 입력 이미지를 업로드해주세요."

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        steps = int(num_inference_steps)
        out_w = int(width)
        out_h = int(height)
        max_len = int(max_sequence_length)
        start_time = time.time()

        # Convert input image to PIL if needed
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)
        orig_w, orig_h = input_image.size

        # Resize input image to target output dimensions
        if orig_w != out_w or orig_h != out_h:
            input_image = input_image.resize((out_w, out_h), Image.LANCZOS)
            print(f"입력 이미지 크기: {orig_w}x{orig_h} → 리사이즈: {out_w}x{out_h}")
        else:
            print(f"입력 이미지 크기: {orig_w}x{orig_h}")
        print(f"추론 스텝: {steps}, 시드: {int(seed)}, 강도: {strength}")
        print(
            f"이미지 생성 중... (steps: {steps}, guidance: {guidance_scale}, true_cfg: {true_cfg_scale})"
        )
        if negative_prompt:
            print(f"부정 프롬프트: {negative_prompt[:50]}...")

        progress(0.0, desc="이미지 생성 준비 중...")
        progress(0.05, desc="추론 시작...")
        print("추론 시작...")

        # Callback to report each inference step to Gradio progress bar and CLI status bar
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.85
            progress(
                progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)"
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
            print(f"\r{line:<80}", end="", flush=True)
            if current == steps:
                print()
            return callback_kwargs

        # Build pipeline arguments
        pipe_kwargs = {
            "image": input_image,
            "prompt": prompt,
            "strength": float(strength),
            "num_inference_steps": steps,
            "guidance_scale": float(guidance_scale),
            "max_sequence_length": max_len,
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        # Add negative prompt if provided and true_cfg_scale > 1
        if negative_prompt and true_cfg_scale > 1.0:
            pipe_kwargs["negative_prompt"] = negative_prompt
            pipe_kwargs["true_cfg_scale"] = float(true_cfg_scale)

        image = pipe(**pipe_kwargs).images[0]

        elapsed = time.time() - start_time
        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]

        output_path = f"{base_name}_{timestamp}_{DEVICE}_step{steps}_cfg{guidance_scale}_tcfg{true_cfg_scale}_str{strength}_{out_w}x{out_h}_seed{int(seed)}.png"
        image.save(output_path)
        print(f"이미지 저장됨: {output_path}")
        print(f"총 소요 시간: {elapsed:.1f}초")

        progress(1.0, desc="완료!")
        return image, f"✓ 이미지 생성 완료! ({elapsed:.1f}초) 저장됨: {output_path}"

    except Exception as e:
        return None, f"오류: {str(e)}"


def main():
    global demo

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Load model once at startup
    load_model()

    # Create Gradio interface
    with gr.Blocks(title="Flux.1 Dev Uncensored Image-to-Image") as demo:
        gr.Markdown("# Flux.1 Dev Uncensored Image-to-Image")
        gr.Markdown(
            f"입력 이미지와 프롬프트를 사용하여 이미지를 변환합니다. (Device: **{DEVICE.upper()}**)"
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="입력 이미지",
                    type="pil",
                    value="Test01.png",
                    height=600,
                )
                prompt_input = gr.Textbox(
                    label="프롬프트 (Prompt)",
                    info="변환할 이미지를 설명하세요 (최대 512 토큰)",
                    placeholder="예: a beautiful woman standing on the beach",
                    value=DEFAULT_PROMPT,
                    lines=3,
                )
                negative_prompt_input = gr.Textbox(
                    label="부정 프롬프트 (Negative Prompt)",
                    info="이미지에 포함되지 않기를 원하는 요소 (True CFG Scale > 1 필요)",
                    placeholder="예: blurry, bad quality, distorted",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                )

                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        strength_input = gr.Slider(
                            label="강도 (Strength)",
                            info="입력 이미지 변환 정도 (0.0: 변환 없음, 1.0: 완전히 새로 생성)",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.75,
                        )
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=28,
                        )

                    with gr.Row():
                        width_input = gr.Slider(
                            label="너비 (Width)",
                            info="출력 이미지의 너비 (이미지 업로드 시 자동 설정)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=768,
                        )
                        height_input = gr.Slider(
                            label="높이 (Height)",
                            info="출력 이미지의 높이 (이미지 업로드 시 자동 설정)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1024,
                        )

                    with gr.Row():
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=0.0,
                            maximum=20.0,
                            step=0.1,
                            value=3.5,
                        )
                        true_cfg_input = gr.Slider(
                            label="True CFG Scale",
                            info=">1 에서 부정 프롬프트와 함께 CFG 활성화",
                            minimum=1.0,
                            maximum=20.0,
                            step=0.1,
                            value=1.0,
                        )

                    with gr.Row():
                        max_seq_input = gr.Slider(
                            label="최대 시퀀스 길이",
                            info="프롬프트의 최대 토큰 수",
                            minimum=64,
                            maximum=512,
                            step=64,
                            value=512,
                        )
                        seed_input = gr.Slider(
                            label="시드 (Seed)",
                            info="재현성을 위한 난수 시드 (-1: 랜덤)",
                            minimum=-1,
                            maximum=1000,
                            step=1,
                            value=42,
                        )

                submit_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column():
                image_output = gr.Image(label="생성된 이미지", height=800)
                status_output = gr.Textbox(label="상태", interactive=False)

        # Auto-update width/height sliders when image is uploaded
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[width_input, height_input],
        )

        # Connect button to generation function
        submit_btn.click(
            fn=generate_image,
            inputs=[
                image_input,
                prompt_input,
                negative_prompt_input,
                strength_input,
                width_input,
                height_input,
                steps_input,
                guidance_input,
                true_cfg_input,
                max_seq_input,
                seed_input,
            ],
            outputs=[image_output, status_output],
        )

    # Launch the interface
    demo.launch(inbrowser=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
