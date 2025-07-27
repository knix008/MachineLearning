import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import gradio as gr

# 텐서 비우기
torch.cuda.empty_cache()

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)

# Use CPU RAM for model weights
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()


def resize_to_multiple_of_16(image):
    w, h = image.size
    # 16의 배수로 맞춤 (비율 유지)
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    # 최소 16 보장
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    return image.resize((new_w, new_h), Image.LANCZOS)


def generate(input_image, prompt, num_inference_steps, strength, guidance_scale):
    if input_image is None or prompt.strip() == "":
        return None, "입력 이미지 또는 프롬프트가 없습니다."
    input_image = input_image.convert("RGB")
    orig_w, orig_h = input_image.size
    input_image = resize_to_multiple_of_16(input_image)
    resized_w, resized_h = input_image.size
    with torch.no_grad():
        result = pipe(
            prompt,
            image=input_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
        ).images[0]
    out_w, out_h = result.size
    description = (
        f"프롬프트: {prompt}\n"
        f"입력 이미지 크기: {orig_w}x{orig_h}\n"
        f"리사이즈된 입력 크기: {resized_w}x{resized_h}\n"
        f"출력 이미지 크기: {out_w}x{out_h}\n"
        f"Inference Steps: {num_inference_steps}, Strength: {strength}, Guidance Scale: {guidance_scale}"
    )
    torch.cuda.empty_cache()  # 메모리 정리 (CUDA가 다룬되느 일이 종종 발생했다.)
    return result, description


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(
            label="입력 이미지 (이미지 변환의 기준이 되는 입력 이미지입니다.)",
            type="pil",
            height=500,
        ),
        gr.Textbox(
            label="프롬프트 (생성될 이미지의 내용을 설명하는 텍스트입니다.)",
            value="visible pores, highly detailed, 8k, high quality, ultra high definition, masterpiece, photorealistic, cinematic lighting, intricate details",
        ),
        gr.Slider(
            1, 50, value=28, step=1, label="Inference Steps (이미지 생성 반복 횟수, 높을수록 품질↑, 속도↓)"
        ),
        gr.Slider(
            0.0, 1.0, value=0.5, step=0.01, label="Strength (원본 이미지 변형 강도, 높을수록 원본과 달라짐)"
        ),
        gr.Slider(
            0.0, 10.0, value=5.0, step=0.01, label="Guidance Scale (프롬프트 반영 강도, 높을수록 프롬프트 영향↑)"
        ),
    ],
    outputs=[
        gr.Image(label="결과 이미지 (생성된 결과 이미지입니다.)", height=500),
        gr.Textbox(label="이미지 설명 (생성 과정 및 파라미터 정보가 표시됩니다.)"),
    ],
    title="SDXL Turbo 이미지 변환",
    description="각 항목의 설명을 참고하여 이미지를 생성하세요.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
