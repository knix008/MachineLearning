import gradio as gr
import torch
from diffusers import AutoPipelineForImage2Image

# GPU 사용 가능 여부 확인 및 장치 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 사전 훈련된 Stable Diffusion Image-to-Image 모델 불러오기
# torch_dtype=torch.float16은 GPU 사용 시 메모리를 절약해줍니다.
pipe = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 이미지 생성 함수 정의
def generate_similar_image(input_image, prompt, strength, guidance_scale, negative_prompt):
    """
    입력된 이미지와 프롬프트를 기반으로 유사한 이미지를 생성하는 함수

    Args:
        input_image (PIL.Image): 사용자가 업로드한 원본 이미지
        prompt (str): 긍정 프롬프트 (생성하고 싶은 이미지의 특징)
        strength (float): 원본 이미지와의 유사도 (0에 가까울수록 원본과 유사, 1에 가까울수록 프롬프트 영향을 많이 받음)
        guidance_scale (float): 프롬프트 충실도 (값이 높을수록 프롬프트를 더 따름)
        negative_prompt (str): 부정 프롬프트 (피하고 싶은 요소 설명)

    Returns:
        PIL.Image: 생성된 이미지
    """
    if input_image is None:
        raise gr.Error("이미지를 먼저 업로드해주세요!")

    print(f"Generating image with prompt: {prompt}")
    print(f"Strength: {strength}, Guidance Scale: {guidance_scale}")

    # 입력 이미지를 RGB로 변환
    init_image = input_image.convert("RGB")

    # 파이프라인을 사용하여 이미지 생성
    generator = torch.Generator(device=device).manual_seed(42)
    
    image = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator
    ).images[0]

    return image

# Gradio 인터페이스 생성
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🖼️ 이미지 기반 유사 이미지 생성기 (Image-to-Image)")
    gr.Markdown("원본 이미지를 업로드하고, 설명을 추가하여 새로운 이미지를 만들어보세요. '유사도'를 조절하여 원본의 형태를 얼마나 유지할지 결정할 수 있습니다.")

    with gr.Row():
        with gr.Column(scale=1):
            # 사용자 입력 컴포넌트
            image_input = gr.Image(type="pil", label="원본 이미지 업로드")
            prompt_input = gr.Textbox(label="프롬프트 (Prompt)", placeholder="예: a modern university logo, shield, letter S")
            negative_prompt_input = gr.Textbox(label="제외할 내용 (Negative Prompt)", placeholder="예: blurry, text, watermark, ugly")
            
            strength_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.75, step=0.05,
                label="유사도 (Strength)",
                info="값이 낮을수록 원본과 비슷해지고, 높을수록 프롬프트의 영향을 많이 받습니다."
            )
            guidance_slider = gr.Slider(
                minimum=1, maximum=20, value=8.0, step=0.5,
                label="프롬프트 충실도 (Guidance Scale)"
            )
            
            generate_button = gr.Button("✨ 이미지 생성하기", variant="primary")

        with gr.Column(scale=1):
            # 결과 출력 컴포넌트
            image_output = gr.Image(label="생성된 이미지")

    # 버튼 클릭 시 함수 실행
    generate_button.click(
        fn=generate_similar_image,
        inputs=[image_input, prompt_input, strength_slider, guidance_slider, negative_prompt_input],
        outputs=image_output,
        api_name="generate" # API로 사용할 경우 이름 지정
    )

# Gradio 앱 실행
if __name__ == "__main__":
    demo.launch()