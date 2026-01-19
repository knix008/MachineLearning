from transformers import logging

logging.set_verbosity_error()

import torch
import gradio as gr
from diffusers import FluxImg2ImgPipeline
from datetime import datetime
from PIL import Image

DEFAULT_IMAGE_PATH = "default.jpg"

device = "cpu"
dtype = torch.float32

# Load image-to-image pipeline (FLUX.1-dev supports img2img, FLUX.2-klein does not)
pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
)
pipe = pipe.to(device)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")


def generate_image(
    input_image, prompt, strength, guidance_scale, num_inference_steps, seed
):
    if input_image is None:
        input_image = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")

    generator = torch.Generator(device=device).manual_seed(int(seed))

    output_image = pipe(
        prompt=prompt,
        image=input_image,
        width=input_image.width,
        height=input_image.height,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        generator=generator,
    ).images[0]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"flux-dev-img2img_{timestamp}.png"
    output_image.save(filename)

    return output_image, f"이미지가 저장되었습니다: {filename}"


with gr.Blocks(title="FLUX.1 Image-to-Image") as demo:
    gr.Markdown("# FLUX.1 Image-to-Image Generator")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", label="Input Image", value=DEFAULT_IMAGE_PATH, height=700
            )
            prompt = gr.Textbox(
                label="Prompt",
                value="8k resolution quality, high detail, high quality, best quality, realistic, masterpiece, cinematic lighting, wearing a dark blue bikini, move her face to make her see the viewer.",
                lines=3,
            )

            with gr.Row():
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    label="Strength (how much to change the input image)",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale",
                )

            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=50, value=28, step=1, label="Inference Steps"
                )
                seed = gr.Number(value=100, label="Seed", precision=0)

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image", height=700)
            status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image,
            prompt,
            strength,
            guidance_scale,
            num_inference_steps,
            seed,
        ],
        outputs=[output_image, status_text],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
