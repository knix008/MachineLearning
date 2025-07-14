# 1. 필수 라이브러리 설치
# !pip install gradio diffusers transformers torch accelerate safetensors

import gradio as gr
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

# 2. Hugging Face Hub 로그인 (최초 1회 필요)
# 터미널에서 아래 명령어를 실행하여 Hugging Face에 로그인하세요.
# huggingface-cli login
# SD3 모델은 사용 전 동의가 필요하므로, 모델 페이지(https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)를 방문하여 접근 권한을 요청해야 합니다.
access_token = "Enter your Hugging Face access token here"
from huggingface_hub import login
login(access_token)

# 3. 모델 로드
# GPU 사용 가능 여부를 확인하고 장치를 설정합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 메모리 효율을 위해 float16 데이터 타입을 사용합니다.
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

try:
    # Stable Diffusion 3 이미지-투-이미지 파이프라인을 로드합니다.
    pipe =  StableDiffusion3Img2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )

    # 모델을 GPU로 이동시킵니다.
    pipe = pipe.to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"모델 로딩 중 오류가 발생했습니다: {e}")
    print("Hugging Face Hub에 로그인했고 모델 접근 권한이 있는지 확인하세요.")
    pipe = None


# 4. 이미지 생성 함수 정의
def generate_image(input_image, prompt, strength=0.8, guidance_scale=7.0):
    """
    입력 이미지와 프롬프트를 기반으로 새로운 이미지를 생성합니다.

    Args:
        input_image (PIL.Image): 사용자가 업로드한 원본 이미지
        prompt (str): 이미지 변경을 위한 텍스트 프롬프트
        strength (float): 원본 이미지의 구조를 얼마나 유지할지에 대한 강도 (0.0 ~ 1.0)
        guidance_scale (float): 프롬프트를 얼마나 따를지에 대한 강도

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

        # 파이프라인을 실행하여 이미지 생성
        generated_image = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=28,
        ).images[0]

        return generated_image
    except Exception as e:
        print(f"이미지 생성 중 오류 발생: {e}")
        raise gr.Error(f"이미지 생성에 실패했습니다. 오류: {e}")


# 5. Gradio 인터페이스 생성
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 Stable Diffusion 3: 이미지 변환기
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
            generate_button = gr.Button("✨ 이미지 생성 ✨", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="🎉 결과 이미지")

    generate_button.click(
        fn=generate_image,
        inputs=[image_input, prompt_input, strength_slider, guidance_slider],
        outputs=image_output,
    )

# 6. Gradio 앱 실행
if __name__ == "__main__":
    demo.launch(debug=True)
