import torch
from diffusers import StableDiffusion3Pipeline
import os
import gradio as gr

#access_token = ""  # 필요시 Hugging Face 토큰 입력

model_path = "stabilityai/stable-diffusion-3.5-medium"

try:
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
except Exception as e:
    print(f"Error loading model: {e}")
    pipe = None

def generate_image(prompt, negative_prompt="", steps=28, guidance=7.0):
    if pipe is None:
        return "Error: Model not loaded. Please check the model path or access token."
    try:
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        )
        image = result.images[0]
        return image
    except Exception as e:
        return f"Error: {e}"


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="프롬프트", value="A cat holding a sign that says hello world"
        ),
        gr.Textbox(label="네거티브 프롬프트", value=""),
        gr.Slider(minimum=1, maximum=50, step=1, value=28, label="Inference Steps"),
        gr.Slider(
            minimum=1.0, maximum=15.0, step=0.1, value=7.0, label="Guidance Scale"
        ),
    ],
    outputs=gr.Image(type="pil", label="생성된 이미지"),
    title="Stable Diffusion 3.5 Gradio Demo",
    description="프롬프트를 입력하고 이미지를 생성하세요.",
)

if __name__ == "__main__":
    demo.launch()
