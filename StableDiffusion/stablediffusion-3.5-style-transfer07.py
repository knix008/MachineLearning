import gradio as gr
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from PIL import Image

# 2. Hugging Face Hub 로그인 (최초 1회 필요)
# huggingface-cli login
# SD3 모델은 사용 전 동의가 필요하므로, 모델 페이지(https://huggingface.co/stabilityai/stable-diffusion-3-medium)를 방문하여 접근 권한을 요청해야 합니다.

# 3. 모델 로드
try:
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    pipe = pipe.to("cpu")
    print("Model loaded successfully.")
except Exception as e:
    print(f"모델 로딩 중 오류가 발생했습니다: {e}")
    print("Hugging Face Hub에 로그인했고 모델 접근 권한이 있는지 확인하세요.")
    pipe = None

# 4. 이미지 생성 함수 정의
def generate_image(
    input_image, 
    prompt, 
    strength=0.8, 
    guidance_scale=7.0, 
    num_inference_steps=28, 
    negative_prompt="", 
    seed=-1
):
    """
    입력 이미지와 프롬프트를 기반으로 새로운 이미지를 생성합니다.

    Args:
        input_image (PIL.Image): 사용자가 업로드한 원본 이미지
        prompt (str): 이미지 변경을 위한 텍스트 프롬프트
        strength (float): 원본 이미지의 구조를 얼마나 유지할지에 대한 강도 (0.0 ~ 1.0)
        guidance_scale (float): 프롬프트를 얼마나 따를지에 대한 강도
        num_inference_steps (int): 이미지 생성시 사용할 추론 단계 수
        negative_prompt (str): 생성에서 피하고 싶은 요소
        seed (int): 랜덤 시드 (-1이면 랜덤)

    Returns:
        PIL.Image: 생성된 이미지
    """
    if pipe is None:
        raise gr.Error(
            "모델이 제대로 로드되지 않았습니다. 프로그램을 재시작하고 오류 메시지를 확인하세요."
        )
    if input_image is None:
        raise gr.Error("이미지를 먼저 업로드해주세요.")
    if not prompt:
        raise gr.Error("프롬프트를 입력해주세요.")

    try:
        # 입력 이미지를 PIL Image 객체로 변환
        init_image = Image.fromarray(input_image).convert("RGB")
        width, height = init_image.size

        # 최대 크기 및 16의 배수로 맞추기
        max_size = 1024
        def round_to_16(x):
            return max(16, (x // 16) * 16)

        if width > max_size or height > max_size:
            if width >= height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            width, height = new_width, new_height

        width = round_to_16(width)
        height = round_to_16(height)
        init_image = init_image.resize((width, height), Image.LANCZOS)

        # 시드 설정
        generator = None
        if seed is not None and int(seed) != -1:
            generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

        # 파이프라인을 실행하여 이미지 생성
        generated_image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        return generated_image
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        raise gr.Error(f"이미지 생성에 실패했습니다. 오류: {e}")

# 5. Gradio 인터페이스 생성
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Stable Diffusion 3.5 Medium : 이미지 변환기
        이미지를 업로드하고, 어떻게 바꿀지 프롬프트로 알려주세요!
        """
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="🖼️ 원본 이미지")
            prompt_input = gr.Textbox(
                label="📝 프롬프트",
                placeholder="예: A vibrant oil painting of a futuristic city",
            )
            negative_prompt_input = gr.Textbox(
                label="🚫 네거티브 프롬프트 (선택)",
                placeholder="예: blurry, low quality, watermark",
                info="생성에서 피하고 싶은 요소를 입력하세요.",
            )
            strength_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.8,
                label="💪 원본 유지 강도",
                info="값이 높을수록 원본 이미지와 유사해집니다.",
            )
            guidance_slider = gr.Slider(
                minimum=0.0,
                maximum=15.0,
                value=7.0,
                label="🧭 프롬프트 충실도",
                info="값이 높을수록 프롬프트를 더 엄격하게 따릅니다.",
            )
            steps_slider = gr.Slider(
                minimum=1,
                maximum=100,
                value=28,
                step=1,
                label="🔁 Inference Steps",
                info="이미지 생성 품질을 높이고 싶다면 값을 올려보세요. (기본값: 28)",
            )
            seed_slider = gr.Slider(
                minimum=-1,
                maximum=2**32-1,
                value=-1,
                step=1,
                label="🌱 Seed (고정값, -1은 랜덤)",
                info="같은 시드로 같은 결과를 얻을 수 있습니다. -1은 매번 랜덤 시드 사용.",
            )
            generate_button = gr.Button("✨ 이미지 생성 ✨", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="🎉 결과 이미지")

    generate_button.click(
        fn=generate_image,
        inputs=[
            image_input, 
            prompt_input, 
            strength_slider, 
            guidance_slider, 
            steps_slider, 
            negative_prompt_input, 
            seed_slider
        ],
        outputs=image_output,
    )
# 6. Gradio 앱 실행
if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True))
