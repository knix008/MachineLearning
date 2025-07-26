import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import gradio as gr

MODEL_ID = "stabilityai/stable-diffusion-3-5-turbo-large"

# 파이프라인 로드 및 메모리 최적화 옵션
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
pipe = pipe.to("cpu")

# 모델 최적화 설정
pipe.enable_model_cpu_offload()  # CPU 메모리 최적화
pipe.enable_sequential_cpu_offload()  # CPU 메모리 최적화
pipe.enable_attention_slicing(1)  # Attention slicing for memory efficiency


def resize_to_multiple_of_16(img, max_size=768):
    w, h = img.size
    aspect = w / h
    # 최대 길이 축 기준 리사이즈
    if aspect >= 1:
        new_w = min(max_size, w)
        new_h = int(new_w / aspect)
    else:
        new_h = min(max_size, h)
        new_w = int(new_h * aspect)
    # 16의 배수로 반올림
    new_w = max(16, int(round(new_w / 16)) * 16)
    new_h = max(16, int(round(new_h / 16)) * 16)
    return img.resize((new_w, new_h), Image.LANCZOS)


def img2img_gradio(input_image, prompt, strength, guidance_scale, steps):
    # 이미지 준비
    if isinstance(input_image, str):
        input_image = Image.open(input_image).convert("RGB")
    else:
        input_image = input_image.convert("RGB")
    input_image = resize_to_multiple_of_16(
        input_image, max_size=768
    )  # 메모리 절약 위해 크기 축소

    # torch.no_grad로 불필요한 메모리 사용 방지
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
        )
    # CPU 메모리 해제
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result.images[0]


with gr.Blocks() as demo:
    gr.Markdown(
        "# Stable Diffusion 3.5 Large Turbo - Image to Image (CPU, 메모리 최적화)"
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="원본 이미지", type="pil", width=500)
            prompt = gr.Textbox(
                label="프롬프트", value="A futuristic cityscape with vibrant colors"
            )
            strength = gr.Slider(
                label="변환 강도 (strength)",
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.05,
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5
            )
            steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=50, value=30, step=1
            )
            run_btn = gr.Button("이미지 변환 실행")
        with gr.Column():
            output_image = gr.Image(label="결과 이미지")

    run_btn.click(
        fn=img2img_gradio,
        inputs=[input_image, prompt, strength, guidance_scale, steps],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
