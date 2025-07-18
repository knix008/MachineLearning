import torch
from diffusers import FluxPipeline
import gradio as gr
import time
from PIL import Image

#access_token = ""

#from huggingface_hub import login
#login(access_token)

# 모델 로딩 (최초 한 번만)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload() # This can make a warning message like "add_prefix_space" ...


def generate_image(prompt):
    start_time = time.time()
    generator = torch.Generator("cpu").manual_seed(0) # For reproducibility, how about using "cuda"?
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator,
    ).images[0]
    elapsed = time.time() - start_time
    return image, f"걸린 시간: {elapsed:.2f}초"


examples = [
    ["A cat holding a sign that says hello world"],
    ["A dog riding a bicycle"],
]

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="프롬프트를 입력하세요"),
    outputs=[gr.Image(label="생성된 이미지"), gr.Textbox(label="걸린 시간 (초)")],
    title="FLUX Schnell 이미지 생성기",
    examples=examples,
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()
