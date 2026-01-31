import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import os
import gradio as gr
import math
import numpy as np
import gc
import signal
import sys
import platform

model_id = "black-forest-labs/FLUX.2-klein-4B"

DEFAULT_PROMPT = (
    "she is wearing the bikini, the beach wide sun cap, and  the sunglasses."
)

# Global variables for model
pipe = None
demo = None
DEVICE = None
DTYPE = None


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_dtype(device):
    """Get the appropriate dtype for the device."""
    if device == "cpu":
        return torch.float32
    else:
        return torch.bfloat16


def print_system_info():
    """Print system hardware and device information."""
    print("=" * 60)
    print("시스템 정보")
    print("=" * 60)

    # CPU/OS 정보
    print(f"CPU: {platform.processor()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # 디바이스 정보
    print("-" * 60)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print("디바이스: CUDA")
        print(f"GPU: {device_name}")
        print(f"GPU 개수: {device_count}")
        print(f"VRAM: {total_memory:.2f} GB")
        print(f"CUDA 버전: {torch.version.cuda}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("디바이스: MPS (Apple Silicon)")
    else:
        print("디바이스: CPU")
        print("경고: GPU를 사용할 수 없습니다. CPU로 실행됩니다.")

    print(f"데이터 타입: {DTYPE}")
    print("=" * 60)


def cleanup():
    """Release all resources."""
    global pipe, demo
    print("\n자원 해제 중...")
    try:
        if pipe is not None:
            del pipe
        if demo is not None:
            demo.close()
        # Clear GPU cache based on device type
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print("자원 해제 완료!")
    except Exception as e:
        print(f"자원 해제 중 오류: {e}")


def signal_handler(_sig, _frame):
    """Handle keyboard interrupt signal."""
    print("\n\nKeyboard Interrupt 감지!")
    cleanup()
    sys.exit(0)


def load_model():
    """Load and initialize the Flux2 Img2Img model."""
    global pipe, DEVICE, DTYPE

    # Detect device and dtype
    DEVICE = get_device()
    DTYPE = get_dtype(DEVICE)

    print(f"\n사용할 디바이스: {DEVICE.upper()}")
    print(f"데이터 타입: {DTYPE}")
    print("모델 로딩 중...")

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id, torch_dtype=DTYPE
    )

    # Device-specific setup
    if DEVICE == "cuda" or DEVICE == "cpu":
        # CUDA: Use CPU offload for memory optimization
        pipe = pipe.to(DEVICE)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print("메모리 최적화 적용됨 (CUDA)")
    elif DEVICE == "mps":
        # MPS: Direct device placement, no memory optimization
        pipe = pipe.to(DEVICE)
        print("MPS 디바이스에 모델 로드됨")
    else:
        print("경고: 지원되지 않는 디바이스입니다!!!")
        exit(1)

    print("모델 로딩 완료!")
    return pipe


def create_image_grid(images, grid_size=None):
    """Create a grid/collage from multiple images."""
    if not images:
        return None

    n_images = len(images)

    if grid_size is None:
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)
    else:
        cols, rows = grid_size

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    resized_images = []
    for img in images:
        resized = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
        resized_images.append(resized)

    grid_width = cols * max_width
    grid_height = rows * max_height
    grid_image = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))

    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * max_height
        grid_image.paste(img, (x, y))

    return grid_image


def blend_images(images, mode="average"):
    """Blend multiple images together."""
    if not images:
        return None

    if len(images) == 1:
        return images[0]

    target_width = images[0].width
    target_height = images[0].height

    resized_images = [
        img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        for img in images
    ]

    if mode == "average":
        arrays = [np.array(img, dtype=np.float32) for img in resized_images]
        avg_array = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(avg_array)

    elif mode == "overlay":
        result = resized_images[0].copy()
        alpha = 1.0 / len(resized_images)
        for img in resized_images[1:]:
            result = Image.blend(result, img, alpha)
        return result

    else:  # multiply
        arrays = [np.array(img, dtype=np.float32) / 255.0 for img in resized_images]
        result = arrays[0]
        for arr in arrays[1:]:
            result = result * arr
        result = (result * 255).astype(np.uint8)
        return Image.fromarray(result)


def generate_image(
    img1,
    img2,
    img3,
    img4,
    prompt,
    combine_mode,
    height,
    width,
    guidance_scale,
    num_inference_steps,
    seed,
):
    """Generate image from multiple input images and text prompt."""
    global pipe

    # Collect non-None images
    images = []
    for img in [img1, img2, img3, img4]:
        if img is not None:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            images.append(img)

    if len(images) == 0:
        return None, None, "오류: 최소 1개 이상의 이미지를 입력해주세요."

    if not prompt:
        return None, None, "오류: 프롬프트를 입력해주세요."

    try:
        # Combine input images
        print(f"입력 이미지 {len(images)}개 결합 중... (모드: {combine_mode})")

        if combine_mode == "grid":
            combined_image = create_image_grid(images)
        elif combine_mode == "blend_average":
            combined_image = blend_images(images, mode="average")
        elif combine_mode == "blend_overlay":
            combined_image = blend_images(images, mode="overlay")
        elif combine_mode == "blend_multiply":
            combined_image = blend_images(images, mode="multiply")
        else:
            combined_image = create_image_grid(images)

        # Resize combined image for the model
        combined_image = combined_image.resize(
            (width, height), Image.Resampling.LANCZOS
        )
        print(f"결합된 이미지 크기: {combined_image.size}")

        # Setup generator
        generator = torch.Generator(device=DEVICE)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))

        print(f"이미지 생성 중... (steps: {num_inference_steps})")

        # Generate image
        result = pipe(
            prompt=prompt,
            image=combined_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}_seed{int(seed)}.png"
        result.save(output_path)
        print(f"이미지 저장됨: {output_path}")

        return (
            combined_image,
            result,
            f"✓ 이미지 생성 완료! ({len(images)}개 이미지 사용) 저장됨: {output_path}",
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, None, f"오류: {str(e)}"


def main():
    global demo

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Print system info
        print_system_info()

        # Load model once at startup
        load_model()

        # Create Gradio interface
        with gr.Blocks(title="Flux Klein 4B 다중 이미지 생성기") as demo:
            gr.Markdown(f"# Flux Klein 4B Multi-Image Generator ({DEVICE.upper()})")
            gr.Markdown(
                "여러 이미지를 업로드하고, 이를 기반으로 새로운 이미지를 생성합니다."
            )

            # 입력 이미지들
            gr.Markdown("### 입력 이미지 (최대 4개)")
            with gr.Row():
                image_input1 = gr.Image(label="이미지 1", type="pil", height=250)
                image_input2 = gr.Image(label="이미지 2", type="pil", height=250)
                image_input3 = gr.Image(label="이미지 3", type="pil", height=250)
                image_input4 = gr.Image(label="이미지 4", type="pil", height=250)

            # 결합 방식 선택
            with gr.Row():
                combine_mode = gr.Radio(
                    label="이미지 결합 방식",
                    choices=[
                        ("그리드 (Grid)", "grid"),
                        ("평균 블렌드 (Average Blend)", "blend_average"),
                        ("오버레이 블렌드 (Overlay Blend)", "blend_overlay"),
                        ("곱셈 블렌드 (Multiply Blend)", "blend_multiply"),
                    ],
                    value="grid",
                )

            # 프롬프트 입력
            prompt_input = gr.Textbox(
                label="생성 프롬프트 (영어)",
                placeholder="예: Combine these images into a beautiful landscape",
                value=DEFAULT_PROMPT,
                lines=3,
            )

            # 고급 설정
            with gr.Accordion("고급 설정", open=False):
                with gr.Row():
                    height_input = gr.Slider(
                        label="출력 높이", minimum=256, maximum=1024, step=64, value=1024
                    )
                    width_input = gr.Slider(
                        label="출력 너비", minimum=256, maximum=1024, step=64, value=768
                    )

                with gr.Row():
                    guidance_input = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                    )
                    steps_input = gr.Slider(
                        label="추론 스텝", minimum=1, maximum=20, step=1, value=4
                    )

                seed_input = gr.Slider(
                    label="시드 (-1=랜덤)", minimum=-1, maximum=1000, step=1, value=42
                )

            # 생성 버튼
            submit_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            # 결과 표시
            gr.Markdown("### 결과")
            with gr.Row():
                combined_output = gr.Image(label="결합된 입력 이미지", height=500)
                image_output = gr.Image(label="생성된 이미지", height=500)

            status_output = gr.Textbox(label="상태", interactive=False)

            # Connect button to generation function
            submit_btn.click(
                fn=generate_image,
                inputs=[
                    image_input1,
                    image_input2,
                    image_input3,
                    image_input4,
                    prompt_input,
                    combine_mode,
                    height_input,
                    width_input,
                    guidance_input,
                    steps_input,
                    seed_input,
                ],
                outputs=[combined_output, image_output, status_output],
            )

            # 사용 예시
            with gr.Accordion("사용 방법", open=False):
                gr.Markdown(
                    """
## 사용 방법

1. **이미지 업로드**: 최대 4개의 이미지를 업로드합니다.
2. **결합 방식 선택**:
   - **그리드**: 이미지들을 격자 형태로 배치
   - **평균 블렌드**: 모든 이미지의 픽셀을 평균화
   - **오버레이 블렌드**: 이미지들을 순차적으로 오버레이
   - **곱셈 블렌드**: 픽셀 값을 곱하여 어두운 부분 강조
3. **프롬프트 입력**: 원하는 결과를 영어로 설명합니다.
4. **생성 버튼 클릭**: 이미지 생성을 시작합니다.

## 프롬프트 예시

- `Combine these images into a cohesive artistic composition`
- `Merge these portraits into a single person with features from all`
- `Create a seamless landscape by blending these scenes`
                """
                )

            # Launch the interface
            demo.launch(inbrowser=True)

    except KeyboardInterrupt:
        print("\n\nKeyboard Interrupt 감지!")
    except Exception as e:
        print(f"\n오류 발생: {e}")
    finally:
        print("\n프로그램 종료 중...")
        cleanup()
        print("프로그램 종료 완료!")


if __name__ == "__main__":
    main()
