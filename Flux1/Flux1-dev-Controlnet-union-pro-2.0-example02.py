import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
import datetime
import cv2
from PIL import Image
import gradio as gr
import numpy as np

base_model = "black-forest-labs/FLUX.1-dev"
controlnet_model_union = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"

controlnet = FluxControlNetModel.from_pretrained(
    controlnet_model_union, torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)

# 모델을 CPU로 오프로드하여 메모리 사용량을 줄입니다.
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
print("모델 로딩 완료!")


def make_control_image_pil(input_pil, canny_low, canny_high):
    img = cv2.cvtColor(np.array(input_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    # 흑백 이미지를 RGB로 변환
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def generate_image(
    input_image,
    prompt,
    canny_low,
    canny_high,
    conditioning_scale,
    guidance_scale,
    steps,
    seed,
):
    # 16의 배수로 리사이즈 (비율 유지)
    max_side = 1024  # 생성될 이미지의 최대크기(가로 또는 세로)
    orig_w, orig_h = input_image.size
    scale = min(max_side / orig_w, max_side / orig_h, 1.0)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    # 16의 배수로 맞추기
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    resized_image = input_image.resize((new_w, new_h), Image.LANCZOS)

    control_image = make_control_image_pil(resized_image, canny_low, canny_high)
    width, height = control_image.size

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed > 0 else None

    image = pipe(
        prompt,
        control_image=control_image,
        width=width,
        height=height,
        controlnet_conditioning_scale=conditioning_scale,
        control_guidance_end=0.8,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # 이미지 저장
    filename = (
        f"Flux1-Union-Pro-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    )
    image.save(filename)

    return image


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="Input Image", height=500, value="default.jpg"),
        gr.Textbox(
            lines=3,
            label="Prompt",
            value="A young girl stands gracefully at the edge of a serene beach, her long, flowing hair gently tousled by the sea breeze. She wears a soft, pastel-colored dress that complements the tranquil blues and greens of the coastal scenery. The golden hues of the setting sun cast a warm glow on her face, highlighting her serene expression. The background features a vast, azure ocean with gentle waves lapping at the shore, surrounded by distant cliffs and a clear, cloudless sky. The composition emphasizes the girl's serene presence amidst the natural beauty, with a balanced blend of warm and cool tones.",
            info="생성할 이미지의 내용을 설명하는 프롬프트를 입력하세요.",
        ),
        gr.Slider(
            0,
            255,
            value=100,
            step=1,
            label="Canny Low Threshold",
            info="Canny 엣지 감지의 하한값입니다. 낮을수록 더 많은 엣지가 검출됩니다.",
        ),
        gr.Slider(
            0,
            255,
            value=200,
            step=1,
            label="Canny High Threshold",
            info="Canny 엣지 감지의 상한값입니다. 높을수록 강한 엣지만 검출됩니다.",
        ),
        gr.Slider(
            0.0,
            2.0,
            value=0.7,
            step=0.01,
            label="Conditioning Scale",
            info="ControlNet의 조건(컨트롤 이미지) 반영 강도입니다. 높을수록 컨트롤 이미지의 영향을 더 많이 받습니다.",
        ),
        gr.Slider(
            1.0,
            10.0,
            value=3.5,
            step=0.1,
            label="Guidance Scale",
            info="텍스트 프롬프트의 반영 강도입니다. 높을수록 프롬프트의 영향을 더 많이 받습니다.",
        ),
        gr.Slider(
            1,
            100,
            value=30,
            step=1,
            label="Inference Steps",
            info="이미지 생성 과정의 반복 횟수입니다. 높을수록 결과가 더 정교해지지만 느려집니다.",
        ),
        gr.Number(
            value=42,
            label="Seed (0=Random)",
            info="시드 값입니다. 같은 값이면 같은 결과가 재현됩니다. 0이면 무작위 시드가 사용됩니다.",
        ),
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Flux ControlNet Union Pro 2.0 Demo",
    description="이미지를 업로드하고 파라미터를 조절해 Flux ControlNet으로 이미지를 생성합니다.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
