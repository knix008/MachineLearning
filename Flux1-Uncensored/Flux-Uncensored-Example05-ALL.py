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

DEFAULT_PROMPT = "The image is a high-quality, photorealistic cosplay portrait of a Korean girl with a soft, idol aesthetic. Physical Appearance: Face: She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera with her index finger lightly touching her chin. She has no extra fingers, no extra arms. Hair: She has long, straight jet-black hair with thick, straight-cut bangs (fringe) that frame her face. Attire (Blue & White Bunny Theme): Headwear: She wears tall, upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base, accented with a small white bow. Outfit: She is naked and show her full body without any clothing on her body. Accessories: Around her neck is a blue bow tie attached to a white collar. She wears long, white floral lace fingerless sleeves that extend past her elbows, finished with blue cuffs and small black decorative ribbons. Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows. Pose: She is standing up  gracefully in front of the edge of a light-colored, vintage-style bed or cushioned bench. Her body is slightly angled toward the camera, creating a soft and inviting posture. Setting & Background: Location: A bright, high-key studio set designed to look like a clean, airy bedroom. Background: The background is dominated by large windows with white vertical blinds or curtains, allowing soft, diffused natural-looking light to flood the scene. The background is softly blurred (bokeh). Lighting: The lighting is bright, soft, and even, minimizing harsh shadows and giving the skin a glowing, porcelain appearance. 8k resolution, high-key lighting, cinematic soft focus, detailed textures of denim and lace, gravure photography style."

DEFAULT_NEGATIVE_PROMPT = "Extra hands, extra legs, extra feet, extra arms, extra toes, Waist Pleats, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low resolution, normal quality, ((monochrome)), ((grayscale)), skin spots, wet, acnes, skin blemishes, age spot, man boobs, backlight, mutated hands, (poorly drawn hands:1.33), blurry, (bad anatomy:1.21), (bad proportions:1.33), extra limbs, (disfigured:1.33), (more than 2 nipples:1.33), (missing arms:1.33), (extra legs:1.33), (fused fingers:1.61), (too many fingers:1.61), (unclear eyes:1.33), lowers, bad hands, missing fingers, extra digit, (futa:1.1), bad hands, missing fingers, (cleft chin:1.3)"


# a naked woman leaning against a window sill, a stock photo, by Jakob Gauermann,  shutterstock, open shirt, elegant lady with alabaster skin, androgyn beauty, post processed denoised, fully covered in drapes, smooth pink skin, a beautiful korean woman in white, non binary model, wearing white robe, belly button showing.

# Extra hands, extra legs, extra feet, extra arms, extra toes, Waist Pleats, paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), low resolution, normal quality, 3d,cartoon,anime,(deformed eyes, nose, ears, nose),bad anatomy, ugly, big breasts

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
    """Load and initialize the FLUX.1-dev model with uncensored LoRA (T5-XXL only, CLIP disabled)."""
    global pipe

    print("모델 로딩 중...")
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
    negative_prompt,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    true_cfg_scale,
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
        print(
            f"이미지 생성 중... (steps: {steps}, guidance: {guidance_scale}, true_cfg: {true_cfg_scale})"
        )
        if negative_prompt:
            print(f"부정 프롬프트: {negative_prompt[:50]}...")

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
            "true_cfg_scale": float(true_cfg_scale),
            "generator": generator,
            "callback_on_step_end": step_callback,
        }

        # Add negative prompt if provided
        if negative_prompt:
            neg_inputs = pipe.tokenizer_2(
                negative_prompt,
                padding="max_length",
                max_length=int(max_sequence_length),
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                neg_embeds = pipe.text_encoder_2(
                    neg_inputs["input_ids"].to(DEVICE),
                    output_hidden_states=False,
                )[0]
            pipe_kwargs["negative_prompt_embeds"] = neg_embeds.to(dtype=DTYPE)
            pipe_kwargs["negative_pooled_prompt_embeds"] = torch.zeros(
                1, 768, dtype=DTYPE, device=prompt_embeds.device
            )

        image = pipe(**pipe_kwargs).images[0]

        elapsed = time.time() - start_time
        progress(0.95, desc="이미지 저장 중...")

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]

        output_path = f"{base_name}_{timestamp}_{DEVICE}_step{steps}_cfg{guidance_scale}_tcfg{true_cfg_scale}_{int(width)}x{int(height)}_seed{int(seed)}.png"
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
    with gr.Blocks(title="Flux.1 Dev Uncensored 이미지 생성기") as demo:
        gr.Markdown("# Flux.1 Dev Uncensored 이미지 생성기")
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
                negative_prompt_input = gr.Textbox(
                    label="부정 프롬프트 (Negative Prompt)",
                    info="이미지에 포함되지 않기를 원하는 요소 (True CFG Scale > 1 필요)",
                    placeholder="예: blurry, bad quality, distorted",
                    value=DEFAULT_NEGATIVE_PROMPT,
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
                            maximum=100,
                            step=1,
                            value=28,
                        )
                        guidance_input = gr.Slider(
                            label="Guidance Scale",
                            info="프롬프트 충실도 (낮음: 창의적, 높음: 정확)",
                            minimum=0.0,
                            maximum=20.0,
                            step=0.1,
                            value=3.5,
                        )

                    with gr.Row():
                        true_cfg_input = gr.Slider(
                            label="True CFG Scale",
                            info=">1 에서 부정 프롬프트와 함께 CFG 활성화",
                            minimum=1.0,
                            maximum=20.0,
                            step=0.1,
                            value=1.0,
                        )
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
                negative_prompt_input,
                height_input,
                width_input,
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
