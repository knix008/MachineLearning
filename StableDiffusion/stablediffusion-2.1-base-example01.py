import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
import time
from PIL import Image
import os


def generate_high_resolution_image(prompt, resolution=(1024, 1024)):
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, revision="fp16"
    ).to("cuda")

    image = pipe(
        prompt,
        height=resolution[1],
        width=resolution[0],
        num_inference_steps=40,
        guidance_scale=7.5,
    ).images[0]

    return image


def gradio_interface(prompt, width, height, filename):
    start_time = time.time()
    image = generate_high_resolution_image(prompt, resolution=(width, height))
    elapsed = time.time() - start_time
    elapsed_str = f"이미지 생성 소요 시간: {elapsed:.2f}초"

    # 이미지 저장
    if filename:
        # 확장자 없으면 .png를 붙임
        if not (
            filename.lower().endswith(".png")
            or filename.lower().endswith(".jpg")
            or filename.lower().endswith(".jpeg")
        ):
            filename += ".png"
        # 같은 이름의 파일이 있을 경우 덮어쓰지 않으려면 아래 코드 활용 가능
        # i = 1
        # orig_filename = filename
        # while os.path.exists(filename):
        #     filename = f"{os.path.splitext(orig_filename)[0]}_{i}{os.path.splitext(orig_filename)[1]}"
        #     i += 1
        image.save(filename)
        elapsed_str += f"\n이미지가 '{filename}'으로 저장되었습니다."
    else:
        elapsed_str += "\n(파일명이 입력되지 않아 저장되지 않았습니다.)"
    return image, elapsed_str


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion 2.1 고해상도 이미지 생성 및 저장기")
    with gr.Row():
        prompt = gr.Textbox(
            label="프롬프트",
            value="a futuristic cityscape at night, ultra high resolution, photorealistic, 8k",
        )
    with gr.Row():
        width = gr.Slider(512, 2048, value=1024, step=64, label="가로 해상도")
        height = gr.Slider(512, 2048, value=1024, step=64, label="세로 해상도")
    with gr.Row():
        filename = gr.Textbox(
            label="저장 파일명 (예: my_image.png, 비워두면 저장 안함)",
            value="output.png",
        )
    with gr.Row():
        btn = gr.Button("이미지 생성")
    with gr.Row():
        output = gr.Image(label="생성된 이미지", type="pil")
    with gr.Row():
        time_output = gr.Textbox(label="소요 시간 및 저장 정보", interactive=False)

    btn.click(
        fn=gradio_interface,
        inputs=[prompt, width, height, filename],
        outputs=[output, time_output],
    )

if __name__ == "__main__":
    demo.launch()
