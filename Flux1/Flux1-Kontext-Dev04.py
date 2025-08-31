import torch
import gradio as gr
import time
from diffusers import FluxKontextPipeline
from PIL import Image
import datetime

# Load the pipeline (make sure the model is downloaded or available locally)
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev")
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
print("Loading Model is Complete!!!")


def flux1_kontext_dev(
    prompt,
    input_image: Image.Image,
    guidance=2.5,
    num_inference_steps=30,
    seed=-1,
    width=None,
    height=None,
):
    # Seed generator
    generator = None
    if seed is not None and str(seed).strip() != "" and int(seed) != -1:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    # Use custom width/height if provided, else from image
    img_width = width if width else input_image.width
    img_height = height if height else input_image.height

    # Run the pipeline
    result = pipe(
        prompt=prompt,
        image=input_image,
        width=img_width,
        height=img_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    result.save(
        f"Flux1-Kontext-Dev02_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    # 파라미터 정보 문자열 생성
    param_info = f"""
    **Prompt:** {prompt}\n
    **Guidance Scale:** {guidance}\n
    **Num Inference Steps:** {num_inference_steps}\n
    **Seed:** {seed if seed is not None and str(seed).strip() != '' else '-1 (random)'}\n
    **Image Size:** {img_width}x{img_height}
    """
    return result, param_info


with gr.Blocks() as demo:
    gr.Markdown("# Flux1-Kontext-Dev")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="change her face to see the camera",
            )
            input_image = gr.Image(
                label="Input Image", value="default.jpg", type="pil", height=500
            )
            guidance = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=10.0, value=7.5, step=0.1
            )
            num_inference_steps = gr.Slider(
                label="Num Inference Steps", minimum=1, maximum=50, value=30, step=1
            )
            run_btn = gr.Button("Run")
        with gr.Column():
            output_img = gr.Image(label="Output Image", height=500)
            param_info_md = gr.Markdown(label="Parameter Info")

    seed = gr.Textbox(label="Seed (default: -1, random)", value="-1", placeholder="-1 for random")
    width = gr.Number(label="Width (optional)", value=None, precision=0)
    height = gr.Number(label="Height (optional)", value=None, precision=0)

    def run_model(prompt, input_image, guidance, num_inference_steps, seed, width, height):
        output_img, param_info = flux1_kontext_dev(
            prompt, input_image, guidance, num_inference_steps, seed, width, height
        )
        return output_img, param_info

    run_btn.click(
        run_model,
        inputs=[prompt, input_image, guidance, num_inference_steps, seed, width, height],
        outputs=[output_img, param_info_md],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
