import torch
from diffusers import FluxKontextPipeline
from PIL import Image
import gradio as gr
import datetime

# --- 1. 모델 및 파이프라인 설정 ---
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)

# --- 2. LoRA(로라) 모델 불러오기 및 적용 ---
lora_path = "./models/flux-kontext-make-person-real-lora.safetensors"
pipe.load_lora_weights(
    "models/flux-kontext-make-person-real-lora.safetensors",
    adapter_name="PersonReal",
    prefix=None
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
print("Model and LoRA loaded successfully.")


def resize_image(image):
    """
    이미지를 RGB로 변환하고,
    입력 이미지의 가로세로 비율을 유지하면서,
    가로/세로 모두 16의 배수로 맞춤.
    """
    image = image.convert("RGB")
    w, h = image.size
    # 16의 배수로 맞추기 (최대 크기 제한 없음)
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"Resized image to: {new_w}x{new_h}")
    return image


def generate_image(input_image, prompt, guidance_scale, steps, seed):
    """
    Gradio 인터페이스용 추론 함수
    """
    image = resize_image(input_image)
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed > 0 else None
    result = pipe(
        prompt=prompt,
        image=image,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        width=image.width,
        height=image.height,
        generator=generator,
    ).images[0]
    # 파일 저장 (옵션)
    filename = f"result_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    result.save(filename)
    return result


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="Input Image", value="default.jpg", height=500),
        gr.Textbox(
            lines=2,
            label="Prompt",
            value="8k, high detail, high quality, realistic, masterpiece, best quality, dark blue bikini",
            placeholder="Enter your prompt here...",
        ),
        gr.Slider(
            1.0,
            15.0,
            value=7.5,
            step=0.1,
            label="Guidance Scale",
            info="프롬프트의 영향력(높을수록 프롬프트를 더 강하게 반영)",
        ),
        gr.Slider(
            1,
            50,
            value=25,
            step=1,
            label="Inference Steps",
            info="이미지 생성 반복 횟수(높을수록 결과가 더 정교해지지만 느려집니다)",
        ),
        gr.Number(
            value=42,
            label="Seed (0=Random)",
            info="시드 값. 같은 값이면 같은 결과가 재현됩니다. 0이면 무작위",
        ),
    ],
    outputs=gr.Image(type="pil", label="Result Image", height=500),
    title="Flux Kontext LoRA Demo",
    description="이미지를 업로드하고 프롬프트 및 파라미터를 조절해 FLUX Kontext + LoRA로 이미지를 변환합니다.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
