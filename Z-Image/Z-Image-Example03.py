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

# https://prompthero.com/prompt/16c71686861-z-image-turbo-the-image-is-a-high-quality-photorealistic-cosplay-portrait-of-a-young-asian-woman-with-a-soft-idol-aesthetic-physical

DEFAULT_PROMPT = "The image is a high-quality,photorealistic cosplay portrait of a young Asian woman with a soft, idol aesthetic.Physical Appearance: Face: She has a fair,clear complexion.She is wearing striking bright blue contact lenses that contrast with her dark hair.Her expression is innocent and curious,looking directly at the camera with her index finger lightly touching her chin.Hair: She has long,straight jet-black hair with thick,straight-cut bangs (fringe) that frame her face.Attire (Blue & White Bunny Theme): Headwear: She wears tall,upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base,accented with a small white bow.Outfit: She wears a unique blue denim-textured bodysuit.It features a front zipper,silver buttons,and thin silver chains draped across the chest.The sides are constructed from semi-sheer white lace.Accessories: Around her neck is a blue bow tie attached to a white collar.She wears long,white floral lace fingerless sleeves that extend past her elbows,finished with blue cuffs and small black decorative ribbons.Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows.Pose: She is sitting gracefully on the edge of a light-colored,vintage-style bed or cushioned bench.Her body is slightly angled toward the camera,creating a soft and inviting posture.Setting & Background: Location: A bright,high-key studio set designed to look like a clean,airy bedroom.Background: The background is dominated by large windows with white vertical blinds or curtains,allowing soft,diffused natural-looking light to flood the scene.The background is softly blurred (bokeh).Lighting: The lighting is bright,soft,and even,minimizing harsh shadows and giving the skin a glowing,porcelain appearance.8k resolution,high-key lighting,cinematic soft focus,detailed textures of denim and lace,gravure photography style."

DEFAULT_NEGATIVE_PROMPT = "extra hands,extra legs,extra feet,extra arms,Waist Pleats,paintings,sketches,(worst quality:2),(low quality:2),(normal quality:2),lowres,normal quality,((monochrome)),((grayscale)),skin spots,wet,acnes,skin blemishes,age spot,manboobs,backlight,mutated hands,(poorly drawn hands:1.33),blurry,(bad anatomy:1.21),(bad proportions:1.33),extra limbs,(disfigured:1.33),(more than 2 nipples:1.33),(missing arms:1.33),(extra legs:1.33),(fused fingers:1.61),(too many fingers:1.61),(unclear eyes:1.33),lowers,bad hands,missing fingers,extra digit,(futa:1.1),bad hands,missing fingers,(cleft chin:1.3),exposed nipples"

# Global variables
pipe = None
interface = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


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

    gc.collect()
    print("메모리 정리 완료")


def signal_handler(_sig, _frame):
    """Handle keyboard interrupt signal."""
    print("\n\n키보드 인터럽트 감지됨 (Ctrl+C)")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def load_model():
    """Load and initialize the Z-Image model with optimizations."""
    global pipe

    if pipe is not None:
        print("기존 모델 해제 중...")
        del pipe
        pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"모델 로딩 중... (Device: {DEVICE}, dtype: {DTYPE})")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=False,
    )
    pipe.to(DEVICE)

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

        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
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
        filename = f"{script_name}_{timestamp}_{int(width)}x{int(height)}_gs{guidance_scale}_step{steps}_cfgnorm{cfg_normalization}_seed{int(seed)}.png"

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

                prompt = gr.Textbox(
                    label="프롬프트",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    placeholder="이미지에 대한 설명을 입력하세요",
                    info="생성하고 싶은 이미지에 대한 텍스트 설명입니다.",
                )
                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=2,
                    placeholder="원하지 않는 요소를 입력하세요 (선택사항)",
                    info="생성에서 제외할 요소를 기술합니다.",
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
                        value=1.3,
                        info="프롬프트 충실도. 낮을수록 창의적, 높을수록 정확. 권장: 1.3",
                    )
                    num_inference_steps = gr.Slider(
                        label="추론 스텝",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25,
                        info="이미지 생성 단계 수. 높을수록 품질 향상. 권장: 13",
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
                        value=True,
                        info="CFG 정규화 사용 여부.",
                    )

                generate_btn = gr.Button("이미지 생성", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="생성된 이미지", height=800)
                output_message = gr.Textbox(label="상태", interactive=False)

        load_model_btn.click(
            fn=load_model,
            inputs=[],
            outputs=[model_status],
        )

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
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
