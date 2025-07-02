import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr
import time

# Kandinsky 2.2 모델 (mini 계열, 경량, 고해상도 지원)
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt, width, height):
    start = time.time()
    generator = torch.manual_seed(0)
    image = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=16,
        generator=generator
    ).images[0]
    elapsed = time.time() - start
    return image, f"걸린 시간: {elapsed:.2f}초"

iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="프롬프트를 입력하세요"),
        gr.Slider(512, 1024, value=768, step=64, label="가로 해상도"),
        gr.Slider(512, 1024, value=768, step=64, label="세로 해상도"),
    ],
    outputs=[gr.Image(label="생성된 이미지"), gr.Textbox(label="걸린 시간 (초)")],
    title="Kandinsky 2.2-mini 경량 고해상도 이미지 생성기"
)

if __name__ == "__main__":
    iface.launch()