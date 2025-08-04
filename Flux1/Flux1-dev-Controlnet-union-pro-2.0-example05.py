import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
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
    scale_factor,  # 추가
):
    # 16의 배수로 맞추기
    new_w = (input_image.width // 16) * 16
    new_h = (input_image.height // 16) * 16
    resized_image = input_image.resize((new_w, new_h), Image.LANCZOS)

    # 배율 적용
    scaled_w = int(new_w * scale_factor)
    scaled_h = int(new_h * scale_factor)

    resized_image = resized_image.resize((scaled_w, scaled_h), Image.LANCZOS)
    control_image = make_control_image_pil(resized_image, canny_low, canny_high)

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed > 0 else None

    image = pipe(
        prompt,
        control_image=control_image,
        width=scaled_w,
        height=scaled_h,
        controlnet_conditioning_scale=conditioning_scale,
        control_guidance_end=0.8,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # 이미지 저장
    filename = (
        f"flux1-dev-controlnet-union-pro-2.0-example05-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
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
            value="dark blue bikini, bright, 8k, high detail, realistic, detail skin, high quality, masterpiece, best quality, sunny beach, realistic shadows, vibrant colors",
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
        gr.Slider(
            1.0,
            4.0,
            value=1.0,
            step=1.0,
            label="Scale Factor",
            info="출력 이미지의 크기 배율입니다. 1~4배까지 조절할 수 있습니다.",
        ),
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Flux ControlNet Union Pro 2.0 Demo",
    description="이미지를 업로드하고 파라미터를 조절해 Flux ControlNet으로 이미지를 생성합니다.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
