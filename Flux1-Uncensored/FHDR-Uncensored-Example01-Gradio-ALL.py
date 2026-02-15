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
from diffusers import FluxPipeline
import torch
import gradio as gr

warnings.filterwarnings("ignore", message=".*No LoRA keys associated.*")
logging.getLogger("diffusers").setLevel(logging.ERROR)

DEFAULT_PROMPT = "A naked cute Instagram-style skinny but gorgeous young korean girl model standing on the beach photography, full body, perfect anatomy, no extra fingers, no extra toes, no extra arms, no extra legs, high quality, 4k, detailed, realistic, cinematic lighting, photorealistic, masterpiece, best quality, ultra-detailed, intricate details, realistic skin texture, sharp focus, award-winning photography."


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
    """Load and initialize the FHDR Uncensored model with T5-XXL only (CLIP disabled)."""
    global pipe

    print("모델 로딩 중...")
    print("T5-XXL 텍스트 인코더만 사용합니다. (CLIP 비활성화)")
    pipe = FluxPipeline.from_pretrained(
        "kpsss34/FHDR_Uncensored",
        text_encoder=None,
        tokenizer=None,
        torch_dtype=DTYPE,
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


def generate_image(
    prompt,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    max_sequence_length,
    seed,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate image from text prompt using T5-XXL encoding only."""
    global pipe

    if not prompt:
        return None, "오류: 프롬프트를 입력해주세요."

    try:
        generator_device = "cpu" if DEVICE == "mps" else DEVICE
        generator = torch.Generator(device=generator_device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        steps = int(num_inference_steps)
        start_time = time.time()

        print(f"출력 크기: {int(width)}x{int(height)}")
        print(f"추론 스텝: {steps}, 시드: {int(seed)}")
        print(f"이미지 생성 중... (steps: {steps}, guidance: {guidance_scale})")

        # Count tokens before encoding
        token_ids = pipe.tokenizer_2(prompt, truncation=False)["input_ids"]
        num_tokens = len(token_ids)
        max_len = int(max_sequence_length)
        truncated = num_tokens > max_len
        token_info = f"프롬프트 토큰 수: {num_tokens}/{max_len}"
        if truncated:
            token_info += f" (초과! {num_tokens - max_len}개 토큰 잘림)"
        print(token_info)

        progress(0.0, desc=f"프롬프트 인코딩 중... ({num_tokens} 토큰)")
        print("프롬프트 인코딩 중 (T5-XXL)...")

        # Encode prompt using T5 only (CLIP is disabled)
        text_inputs = pipe.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = pipe.text_encoder_2(
                text_inputs["input_ids"].to(DEVICE),
                output_hidden_states=False,
            )[0]
        prompt_embeds = prompt_embeds.to(dtype=DTYPE)

        # Zero pooled embeddings (normally from CLIP, not needed with T5-only)
        pooled_prompt_embeds = torch.zeros(
            1, 768, dtype=DTYPE, device=prompt_embeds.device
        )

        progress(0.05, desc="추론 시작...")
        print("추론 시작...")

        # Callback to report each inference step to Gradio progress bar and CLI
        def step_callback(_pipe, step_index, _timestep, callback_kwargs):
            current = step_index + 1
            elapsed = time.time() - start_time
            ratio = current / steps
            progress_val = 0.05 + ratio * 0.85
            progress(
                progress_val, desc=f"추론 스텝 {current}/{steps} ({elapsed:.1f}초 경과)"
            )
            bar_len = 30
            filled = int(bar_len * ratio)
            bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
            print(
                f"\r  [{bar}] 스텝 {current}/{steps} ({ratio*100:.0f}%) - {elapsed:.1f}초 경과",
                end="",
                flush=True,
            )
            if current == steps:
                print()
            return callback_kwargs

        # Prepare arguments with pre-computed T5 embeddings
        pipe_kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "height": int(height),
            "width": int(width),
            "num_inference_steps": steps,
            "guidance_scale": float(guidance_scale),
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        image = pipe(**pipe_kwargs).images[0]

        elapsed = time.time() - start_time
        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]

        output_path = f"{base_name}_{timestamp}_{DEVICE}_step{steps}_cfg{guidance_scale}_{int(width)}x{int(height)}_seed{int(seed)}.jpg"

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
    with gr.Blocks(title="FHDR Uncensored 이미지 생성기") as demo:
        gr.Markdown("# FHDR Uncensored 이미지 생성기")
        gr.Markdown(f"프롬프트로 이미지를 생성하세요. (Device: **{DEVICE.upper()}**)")

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="프롬프트 (Prompt)",
                    info="생성할 이미지를 설명하세요 (T5-XXL 인코더 사용, 최대 512 토큰)",
                    placeholder="예: a beautiful woman standing on the beach",
                    value=DEFAULT_PROMPT,
                    lines=3,
                )

                with gr.Accordion("고급 설정", open=True):
                    with gr.Row():
                        height_input = gr.Slider(
                            label="높이 (Height)",
                            info="생성할 이미지의 높이 (픽셀)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=1536,
                        )
                        width_input = gr.Slider(
                            label="너비 (Width)",
                            info="생성할 이미지의 너비 (픽셀)",
                            minimum=256,
                            maximum=2048,
                            step=64,
                            value=768,
                        )

                    with gr.Row():
                        steps_input = gr.Slider(
                            label="추론 스텝 (Inference Steps)",
                            info="생성 품질 (높을수록 고품질, 느림)",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=28,
                        )
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=4.0,
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

        # Connect button to generation function
        submit_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                height_input,
                width_input,
                steps_input,
                guidance_input,
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
