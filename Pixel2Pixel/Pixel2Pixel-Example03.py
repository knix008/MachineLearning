import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
import gradio as gr
import datetime

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def infer(image, prompt, num_inference_steps, image_guidance_scale):
    if image is None:
        return None
    images = pipe(
        prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
    ).images
    out_img = images[0]
    filename = (
        f"Pixel2Pixel-Result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    out_img.save(filename)
    return out_img


demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(
            type="pil",
            label="Input Image",
            value="default.jpg",
            height=500,
        ),
        gr.Textbox(
            value="8k, high quality, high detail, best quality, photo realistic",
            label="Prompt",
            info="이미지 변환에 사용할 텍스트 프롬프트를 입력하세요. 예: '고화질, 사실적인 사진'",
        ),
        gr.Slider(
            10,
            50,
            value=28,
            step=1,
            label="Num Inference Steps",
            info="이미지 생성 과정의 반복 횟수 (클수록 품질이 높아지지만 느려집니다).",
        ),
        gr.Slider(
            1,
            10,
            value=1,
            step=0.1,
            label="Image Guidance Scale",
            info="원본 이미지의 반영 정도 (높을수록 원본 이미지에 더 충실하게 변환됨).",
        ),
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Pix2Pix with Stable Diffusion",
    description="이미지 업로드, 프롬프트 입력, 파라미터 조정 후 변환 결과를 확인하세요.",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
