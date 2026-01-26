import torch
from diffusers import Flux2Pipeline
from datetime import datetime
from PIL import Image
import os
import warnings
import gradio as gr
import platform
import psutil

# ======================== 하드웨어 정보 출력 ========================
print("\n" + "=" * 60)
print("🖥️  시스템 정보")
print("=" * 60)

# CPU 정보
cpu_count = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "정보 없음"
print(f"CPU: {platform.processor()}")
print(f"  - 코어: {cpu_count}개 (논리 코어: {cpu_count_logical}개)")
print(f"  - 클럭: {cpu_freq} MHz")

# RAM 정보
ram = psutil.virtual_memory()
print(f"\n메모리 (RAM):")
print(f"  - 총 용량: {ram.total / (1024**3):.2f} GB")
print(f"  - 사용 중: {ram.used / (1024**3):.2f} GB")
print(f"  - 여유: {ram.available / (1024**3):.2f} GB")

# GPU/CUDA 정보
print(f"\n그래픽 카드 (GPU):")
if torch.cuda.is_available():
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - CUDA: 사용 가능 (버전: {torch.version.cuda})")
    print(
        f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
    )
else:
    print(f"  - GPU: 미연결 (CUDA 미지원)")
    print(f"  - VRAM: N/A")

print(f"\n현재 실행: CPU 모드 (GPU VRAM 보조 사용)")
print("=" * 60 + "\n")

# Set device and data type
device = "cpu"
dtype = torch.float32

# Load text-to-image pipeline
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=dtype
).to(device)

# Enable memory optimizations - uses GPU VRAM when available
#if torch.cuda.is_available():
#    print("GPU VRAM 활용 최적화 활성화 중...")
#    pipe.enable_model_cpu_offload()  # 모델 일부를 GPU VRAM에 저장하여 CPU RAM 절약
#    print(f"  → CPU RAM 절약 & GPU VRAM 활용 모드")
#else:
#    print("GPU 미연결 - CPU 전용 최적화 사용")
#    pipe.enable_attention_slicing(1)  # 어텐션 계산 메모리 절약

pipe.enable_model_cpu_offload()  # 모델 일부를 GPU VRAM에 저장하여 CPU RAM 절약
pipe.enable_attention_slicing(1)  # 어텐션 계산 메모리 절약
# pipe.enable_sequential_cpu_offload()  # 시퀀셜 오프로딩으로 메모리 절약(Don't use it with CPU offloading already enabled)
print("모델 로딩 완료!")

prompt_input = "Highly realistic, 4k, high-quality, high resolution, beautiful korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Perfect anatomy. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture,f/1.4 --ar 9:16 --style raw."


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
                placeholder="이미지에 대한 설명을 입력하세요 (77단어 이하 권장)",
                info="생성하고 싶은 이미지에 대한 텍스트 설명입니다. 자세할수록 좋습니다. 예: '여자, 미소, 해변, 빨간 비키니'",
            )

            with gr.Row():
                width = gr.Slider(
                    label="이미지 너비",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=384,
                    info="CPU 환경에서는 384x384 권장 (빠른 생성). 64의 배수여야 합니다.",
                )
                height = gr.Slider(
                    label="이미지 높이",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=384,
                    info="CPU 환경에서는 384x384 권장 (빠른 생성). 64의 배수여야 합니다.",
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
                    value=16,
                    info="이미지 생성 과정의 단계 수입니다. CPU는 10-16 권장 (빠른 생성), GPU는 20-28",
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
