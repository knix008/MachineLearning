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
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"사용 디바이스: {device} | dtype: {dtype}")

def _bytes_to_gb(value_bytes):
    return f"{value_bytes / (1024 ** 3):.2f} GB"

def print_system_resources():
    print("=== 시스템 자원 정보 ===")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU 코어: {os.cpu_count()}")

    if psutil is not None:
        mem = psutil.virtual_memory()
        print(f"RAM: {_bytes_to_gb(mem.available)} / {_bytes_to_gb(mem.total)} (사용 가능/전체)")
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
        print(f"디스크: {_bytes_to_gb(disk.free)} / {_bytes_to_gb(disk.total)} (사용 가능/전체)")
    except Exception:
        print("디스크: 정보를 가져올 수 없습니다.")

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total_vram = _bytes_to_gb(props.total_memory)
            allocated = _bytes_to_gb(torch.cuda.memory_allocated(0))
            reserved = _bytes_to_gb(torch.cuda.memory_reserved(0))
            print(f"CUDA GPU: {props.name} | VRAM: {allocated} (사용중) / {reserved} (예약) / {total_vram} (전체)")
        except Exception:
            print("CUDA GPU: 정보 확인 실패")
    elif torch.backends.mps.is_available():
        print("MPS: 사용 가능 (GPU 메모리 정보는 지원되지 않음)")

print_system_resources()

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
if device == "cuda" or device == "cpu":
    print("Using CUDA or CPU device optimizations...")
    pipe.to(device)
    pipe.enable_model_cpu_offload() # CUDA에서 CPU RAM을 일부 사용
    pipe.enable_attention_slicing() # 안쓰면 GPU 메모리를 더 사용함(속)
    pipe.enable_sequential_cpu_offload() # 안쓰면 CUDA에서 느림
elif device == "mps":
    print("Using MPS device optimizations...")
    pipe.enable_attention_slicing() # 안쓰면 GPU 메모리를 더 사용함(속)
    pipe.enable_vae_slicing() # VAE도 메모리 절약
    pipe.enable_vae_tiling() # VAE도 타일링
    torch.mps.empty_cache()
    # MPS doesn't support cpu_offload well
else:
    print("No valid device found!!!")
    exit(1)

prompt_input = "Highly realistic, 4k, high-quality, high resolution, beautiful full body korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw."

def generate_image(
    prompt, width, height, guidance_scale, num_inference_steps, seed, strength
):
    try:
        # Run the pipeline
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}.png"
        image.save(filename)

        return image, f"✓ 이미지가 저장되었습니다: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Flux.1-dev Image Generator") as interface:
    gr.Markdown("# 🎨 Flux.1-dev Image Generator")
    gr.Markdown("AI를 사용하여 텍스트에서 이미지를 생성합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            # Input parameters
            prompt = gr.Textbox(
                label="프롬프트",
                value=prompt_input,
                lines=3,
                placeholder="이미지에 대한 설명을 입력하세요 (77단어 이하 권장)",
                info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다. 예: '여자, 미소, 해변, 빨간 비키니'",
            )

            with gr.Row():
                width = gr.Slider(
                    label="이미지 너비",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=512,
                    info="생성할 이미지의 너비를 지정합니다 (픽셀). 64의 배수여야 합니다.",
                )
                height = gr.Slider(
                    label="이미지 높이",
                    minimum=256,
                    maximum=1024,
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
                    value=4.0,
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
                    value=0.8,
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
            width,
            height,
            guidance_scale,
            num_inference_steps,
            seed,
            strength,
        ],
        outputs=[output_image, output_message],
    )

    gr.Markdown("---")
    gr.Markdown(
        """
    ### 파라미터 설명:
    
    **프롬프트** (Prompt)
    - 생성하고 싶은 이미지에 대한 텍스트 설명입니다
    - 자세할수록 좋습니다. 예: "여자, 미소, 해변, 빨간 비키니"
    - 77단어 이하 권장
    
    **이미지 크기** (Width/Height)
    - 생성할 이미지의 너비와 높이를 지정합니다
    - 256-1024px 범위에서 64의 배수로 설정
    
    **Guidance Scale (프롬프트 강도)**
    - 모델이 프롬프트를 얼마나 따를지 제어합니다
    - 낮을수록 창의적, 높을수록 프롬프트에 정확합니다
    - 권장값: 4-15
    
    **추론 스텝** (Number of Inference Steps)
    - 이미지 생성 과정의 단계 수입니다
    - 높을수록 품질이 좋지만 시간이 더 걸립니다
    - 권장값: 20-28
    
    **시드** (Seed)
    - 난수 생성의 시작점입니다
    - 같은 시드를 사용하면 같은 결과를 얻습니다
    
    **강도** (Strength)
    - 생성 모델의 강도를 제어합니다
    - 낮을수록 다양한 결과, 높을수록 일관성 있는 결과
    - 범위: 0.1-1.0
    """
    )

# Launch the interface
if __name__ == "__main__":
    interface.launch(inbrowser=True)
