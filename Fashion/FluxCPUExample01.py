import torch
from diffusers import FluxPipeline
import gradio as gr
import time

# 모델 로딩 (CPU 고정, float32)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float32
)
pipe.to("cpu")  # 반드시 CPU에 할당


def generate_image(prompt):
    start_time = time.time()
    generator = torch.Generator("cpu").manual_seed(0)
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator,
    ).images[0]
    elapsed = time.time() - start_time
    return image, f"걸린 시간: {elapsed:.2f}초"


iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="프롬프트를 입력하세요"),
    outputs=[gr.Image(label="생성된 이미지"), gr.Textbox(label="걸린 시간 (초)")],
    title="FLUX Schnell 이미지 생성기",
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()
