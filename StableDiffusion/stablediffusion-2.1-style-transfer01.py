import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image

def load_model():
    """img2img를 지원하는 Stable Diffusion 2-1 모델을 로드합니다."""
    model_id = "stabilityai/stable-diffusion-2-1"

    # 파이프라인 생성
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    )

    # 스케줄러 설정
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # GPU 사용 가능시 GPU로 이동
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("GPU를 사용하여 모델을 로드했습니다.")
    else:
        print("CPU를 사용하여 모델을 로드했습니다.")

    return pipe

def style_transfer(
    input_image,
    style_prompt,
    negative_prompt="",
    strength=0.75,
    guidance_scale=7.5,
    num_inference_steps=20,
    seed=-1,
):
    """
    입력 이미지를 지정된 스타일로 변환합니다.
    """
    try:
        # 모델 로드 (실제 서비스에서는 캐싱 권장)
        pipe = load_model()

        # 시드 설정
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)

        if input_image is None:
            return None, "입력 이미지를 업로드해주세요."

        # 이미지 크기 조정
        max_size = 768
        width, height = input_image.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

        # 이미지 변환
        result = pipe(
            prompt=style_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        output_image = result.images[0]
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return output_image, f"변환 완료! 사용된 시드: {seed}"

    except Exception as e:
        return None, f"오류가 발생했습니다: {str(e)}"

def create_interface():
    """Gradio 인터페이스를 생성합니다."""
    style_examples = [
        "oil painting, masterpiece, detailed, vibrant colors",
        "watercolor painting, soft, dreamy, artistic",
        "anime style, cel shading, vibrant, detailed",
        "photorealistic, cinematic lighting, professional photography",
        "sketch, pencil drawing, black and white, artistic",
        "impressionist painting, brush strokes, colorful",
        "cyberpunk style, neon lights, futuristic",
        "vintage, retro, 1950s style, nostalgic",
        "fantasy art, magical, ethereal, mystical",
        "minimalist, clean, simple, modern design",
    ]

    with gr.Blocks(
        title="Stable Diffusion 2.1 Style Transfer", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("# 🎨 Stable Diffusion 2.1 이미지 스타일 변환")
        gr.Markdown("입력 이미지를 원하는 스타일로 변환해보세요!")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 입력")
                input_image = gr.Image(
                    label="변환할 이미지를 업로드하세요", type="pil", height=300
                )
                style_prompt = gr.Textbox(
                    label="스타일 프롬프트",
                    placeholder="예: oil painting, masterpiece, detailed, vibrant colors",
                    lines=3,
                )
                negative_prompt = gr.Textbox(
                    label="네거티브 프롬프트 (선택사항)",
                    placeholder="예: blurry, low quality, distorted",
                    lines=2,
                )
                gr.Markdown("### ⚙️ 파라미터 조정")
                with gr.Row():
                    strength = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.75,
                        step=0.05,
                        label="변환 강도",
                        info="높을수록 더 많이 변환됩니다",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="가이던스 스케일",
                        info="높을수록 프롬프트를 더 잘 따릅니다",
                    )
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="추론 스텝 수",
                        info="높을수록 품질이 좋아지지만 시간이 오래 걸립니다",
                    )
                    seed = gr.Number(
                        value=-1,
                        label="랜덤 시드",
                        info="-1이면 랜덤 시드를 사용합니다",
                    )
                transform_btn = gr.Button(
                    "🎨 이미지 변환하기", variant="primary", size="lg"
                )
                status_text = gr.Textbox(label="상태", interactive=False, lines=2)

            with gr.Column(scale=1):
                gr.Markdown("## 📤 결과")
                output_image = gr.Image(label="변환된 이미지", height=400)

        gr.Markdown("## 💡 스타일 프롬프트 예시")
        with gr.Row():
            for i, example in enumerate(style_examples[:5]):
                gr.Button(example, size="sm").click(
                    fn=lambda x=example: x, outputs=style_prompt
                )
        with gr.Row():
            for i, example in enumerate(style_examples[5:]):
                gr.Button(example, size="sm").click(
                    fn=lambda x=example: x, outputs=style_prompt
                )

        gr.Markdown(
            """
        ## 📖 사용법

        1. **이미지 업로드**: 변환하고 싶은 이미지를 업로드하세요
        2. **스타일 설정**: 원하는 스타일을 설명하는 프롬프트를 입력하세요
        3. **파라미터 조정**: 변환 강도와 품질을 조정하세요
        4. **변환 실행**: "이미지 변환하기" 버튼을 클릭하세요

        ### 💡 팁
        - **변환 강도**: 0.3-0.7 정도가 적당합니다
        - **가이던스 스케일**: 7-10 정도가 좋은 결과를 보여줍니다
        - **추론 스텝**: 20-30 스텝이 품질과 속도의 균형점입니다
        """
        )

        transform_btn.click(
            fn=style_transfer,
            inputs=[
                input_image,
                style_prompt,
                negative_prompt,
                strength,
                guidance_scale,
                num_inference_steps,
                seed,
            ],
            outputs=[output_image, status_text],
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()