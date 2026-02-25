import torch
from diffusers import Flux2Pipeline
from datetime import datetime
from PIL import Image
import os
import warnings
import gradio as gr
import platform
import shutil
import signal
import sys
import gc
import atexit

try:
    import psutil
except Exception:
    psutil = None

# Set device and data type
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32


prompt_input = "A photorealistic gravure-style full-body portrait of a beautiful young korean woman standing by a large window in a bright white room. She has long dark brown hair and a soft,alluring expression.She is wearing a stylish black lingerie set with mesh details and strappy accents, paired with black fishnet thigh-high stockings. She is standing by a white window seat covered with a white faux fur rug, with one leg tucked under her and the other leg extended down white steps. She leans her elbow on the window sill, touching her hair. The background features sheer white curtains and a blurred city view through the window grid. Bright natural daylight, high-key lighting, realistic skin texture,8k resolution, elegant boudoir aesthetic. Key Stylistic Keywords: High-key lighting, white room,black lingerie, fishnets, window seat, faux fur texture, natural daylight, photorealistic, gravure style, elegant, airy."

print(f"사용 디바이스: {device} | dtype: {dtype}")


def _bytes_to_gb(value_bytes):
    return f"{value_bytes / (1024 ** 3):.2f} GB"


def print_system_resources():
    print("=== 시스템 자원 정보 ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU 코어: {os.cpu_count()}")

    if psutil is not None:
        mem = psutil.virtual_memory()
        print(
            f"RAM: {_bytes_to_gb(mem.available)} / {_bytes_to_gb(mem.total)} (사용 가능/전체)"
        )
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            total_ram = page_size * phys_pages
            print(f"RAM: {_bytes_to_gb(total_ram)} (전체)")
        except Exception:
            print("RAM: 정보를 가져올 수 없습니다.")

    try:
        disk = shutil.disk_usage(os.getcwd())
        print(
            f"디스크: {_bytes_to_gb(disk.free)} / {_bytes_to_gb(disk.total)} (사용 가능/전체)"
        )
    except Exception:
        print("디스크: 정보를 가져올 수 없습니다.")

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total_vram = _bytes_to_gb(props.total_memory)
            allocated = _bytes_to_gb(torch.cuda.memory_allocated(0))
            reserved = _bytes_to_gb(torch.cuda.memory_reserved(0))
            print(
                f"CUDA GPU: {props.name} | VRAM: {allocated} (사용중) / {reserved} (예약) / {total_vram} (전체)"
            )
        except Exception:
            print("CUDA GPU: 정보 확인 실패")
    elif torch.backends.mps.is_available():
        print("MPS: 사용 가능 (GPU 메모리 정보는 지원되지 않음)")


print_system_resources()


def _clear_device_cache():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def cleanup_resources():
    global pipe, interface
    try:
        print("\n[종료] 자원 해제 시작...")
        if "interface" in globals() and interface is not None:
            try:
                interface.close()
            except Exception:
                pass
        if "pipe" in globals() and pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            pipe = None
        _clear_device_cache()
        print("[종료] 자원 해제 완료.")
    except Exception:
        print("[종료] 자원 해제 중 오류 발생.")


def _handle_sigint(signum, frame):
    cleanup_resources()
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_sigint)
atexit.register(cleanup_resources)

# Actually, more RAM is required to run this program. Not working in 32GB. More than 48GB RAM required.
# Load text-to-image pipeline
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=dtype, low_cpu_mem_usage=True
)

# Device-specific pipeline setup
if device == "cuda":
    print("Using CUDA device optimizations...")
    pipe.enable_model_cpu_offload()  # CUDA에서 CPU RAM을 일부 사용
    pipe.enable_attention_slicing()  # 안쓰면 GPU 메모리를 더 사용함(속)
    pipe.enable_sequential_cpu_offload()  # 안쓰면 CUDA에서 느림
elif device == "mps":
    print("Using MPS device optimizations...")
    print("No memory optimizations applied.")
    # MPS doesn't support cpu_offload well
else:
    print("Using CPU device optimizations...")
    pipe.enable_model_cpu_offload()  # CUDA에서 CPU RAM을 일부 사용
    pipe.enable_attention_slicing()  # 안쓰면 GPU 메모리를 더 사용함(속)
    pipe.enable_sequential_cpu_offload()  # 안쓰면 CUDA에서 느림

print("모델 로딩 완료!")


def generate_image(
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    true_cfg_scale,
    num_inference_steps,
    seed,
    strength,
):
    try:
        # Print generation parameters to CLI
        print("=" * 60)
        print("=== 이미지 생성 파라미터 ===")
        print(f"  프롬프트: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        if negative_prompt and negative_prompt.strip() and true_cfg_scale > 1.0:
            print(
                f"  네거티브 프롬프트: {negative_prompt.strip()[:100]}{'...' if len(negative_prompt.strip()) > 100 else ''}"
            )
            print(f"  True CFG Scale: {true_cfg_scale}")
        else:
            print("  네거티브 프롬프트: (비활성 - True CFG Scale이 1.0)")
        print(f"  이미지 크기: {width}x{height}")
        print(f"  Guidance Scale: {guidance_scale}")
        print(f"  추론 스텝: {num_inference_steps}")
        print(f"  시드: {int(seed)}")
        print(f"  강도: {strength}")
        print(f"  디바이스: {device} | dtype: {dtype}")
        print("=" * 60)

        # Build pipeline arguments
        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": torch.Generator(device=device).manual_seed(seed),
        }

        # Add negative prompt when provided and true_cfg_scale > 1
        if negative_prompt and negative_prompt.strip() and true_cfg_scale > 1.0:
            pipe_kwargs["negative_prompt"] = negative_prompt.strip()
            pipe_kwargs["true_cfg_scale"] = true_cfg_scale

        # Run the pipeline
        image = pipe(**pipe_kwargs).images[0]

        # Save with timestamp and parameters
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}_{width}x{height}_gs{guidance_scale}_tcfg{true_cfg_scale}_step{num_inference_steps}_seed{int(seed)}_str{strength}.png"
        image.save(filename)

        print(f"✓ 저장 완료: {filename}")
        return image, f"✓ 이미지가 저장되었습니다: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Flux.2-dev Image Generator") as interface:
    gr.Markdown("# 🎨 Flux.2-dev Image Generator")
    gr.Markdown("AI를 사용하여 텍스트에서 이미지를 생성합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            # Input parameters
            prompt = gr.Textbox(
                label="프롬프트",
                value=prompt_input,
                lines=3,
                placeholder="이미지에 대한 설명을 입력하세요.",
                info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다. 예: '여자, 미소, 해변, 빨간 비키니'",
            )
            negative_prompt = gr.Textbox(
                label="네거티브 프롬프트",
                value="blurry, low quality, deformed, ugly, bad anatomy, disfigured, poorly drawn face, mutation, extra limbs, extra fingers, missing fingers, watermark, text, signature",
                lines=2,
                placeholder="원하지 않는 요소를 입력하세요. 예: 'blurry, low quality, deformed'",
                info="이미지에서 제외하고 싶은 요소를 설명합니다. True CFG Scale이 1.0보다 클 때만 적용됩니다.",
            )

            with gr.Row():
                width = gr.Slider(
                    label="이미지 너비",
                    minimum=256,
                    maximum=1536,
                    step=64,
                    value=768,
                    info="생성할 이미지의 너비를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                )
                height = gr.Slider(
                    label="이미지 높이",
                    minimum=256,
                    maximum=1536,
                    step=64,
                    value=1536,
                    info="생성할 이미지의 높이를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale (프롬프트 강도)",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=4.0,
                    info="모델이 프롬프트를 얼마나 따를지 제어합니다. 낮을수록 창의적, 높을수록 정확합니다. 권장: 4-15",
                )
                true_cfg_scale = gr.Slider(
                    label="True CFG Scale (네거티브 프롬프트 강도)",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=1.0,
                    info="네거티브 프롬프트의 강도입니다. 1.0이면 네거티브 프롬프트가 비활성화됩니다. 권장: 1.5-3.0",
                )

            with gr.Row():
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
                    step=0.01,
                    value=0.85,
                    info="생성 모델의 강도를 제어합니다. 낮을수록 다양한 결과, 높을수록 일관성 있는 결과입니다.",
                )

            generate_btn = gr.Button("🚀 이미지 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(label="생성된 이미지", height=800)
            output_message = gr.Textbox(label="상태", interactive=False)

    # Connect the generate button to the function
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
            guidance_scale,
            true_cfg_scale,
            num_inference_steps,
            seed,
            strength,
        ],
        outputs=[output_image, output_message],
    )

# Launch the interface
if __name__ == "__main__":
    interface.launch(inbrowser=True)
