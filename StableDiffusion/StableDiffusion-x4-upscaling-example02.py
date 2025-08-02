import gradio as gr
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import datetime

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, variant="fp16", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

def upscale_image(input_image, prompt):
    image = input_image.convert("RGB")
    w, h = image.size
    if w >= h:
        new_w = 512
        new_h = int(h * 512 / w)
    else:
        new_h = 512
        new_w = int(w * 512 / h)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    result = pipeline(prompt=prompt, image=image).images[0]
    result.save(f"stablediffusion-x4-upscaling-example02_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    return result

default_prompt = (
    "Ultra high resolution, hight detailed, photo realistic, 8k resolution, high quality, masterpiece, cinematic, award winning, hyper realistic, intricate details, sharp focus, depth of field, volumetric lighting, realistic shadows, high dynamic range, ultra detailed textures"
)

demo = gr.Interface(
    fn=upscale_image,
    inputs=[
        gr.Image(type="pil", label="Input Image", value="default.jpg"),
        gr.Textbox(lines=3, value=default_prompt, label="Prompt")
    ],
    outputs=gr.Image(type="pil", label="Upscaled Image"),
    title="Stable Diffusion x4 Upscaler",
    description="이미지를 업로드하고 프롬프트를 입력하면 Stable Diffusion x4 업스케일러로 고해상도 이미지를 생성합니다."
)

if __name__ == "__main__":
    demo.launch()
