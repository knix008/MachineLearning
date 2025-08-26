from aura_sr import AuraSR
from PIL import Image
import gradio as gr

# 모델 로드
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

def upscale_image(input_image, scale):
    image = input_image.convert("RGB")
    if scale == "4x":
        upscaled = aura_sr.upscale_4x_overlapped(image)
    elif scale == "2x":
        upscaled = aura_sr.upscale_2x(image)
    else:
        upscaled = image
    return upscaled

with gr.Blocks() as demo:
    gr.Markdown("# AuraSR Super Resolution Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=500)
            scale = gr.Radio(["2x", "4x"], value="4x", label="Upscale Factor")
            run_btn = gr.Button("Upscale")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Upscaled Image")
    run_btn.click(upscale_image, inputs=[input_image, scale], outputs=output_image)

if __name__ == "__main__":
    demo.launch(inbrowser=True)