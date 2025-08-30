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
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
print("Loading Model is Complete!!!")


def flux1_kontext_dev(
    prompt,
    input_image: Image.Image,
    guidance=2.5,
    num_inference_steps=30,
):
    # Run the pipeline
    result = pipe(
        prompt=prompt,
        image=input_image,
        width=input_image.width,
        height=input_image.height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        # Add more parameters if needed
    ).images[0]

    result.save(
        f"Flux1-Kontext-Dev01_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    # 파라미터 정보 문자열 생성
    param_info = f"""
    **Prompt:** {prompt}\n
    **Guidance Scale:** {guidance}\n
    **Num Inference Steps:** {num_inference_steps}\n
    **Input Image Size:** {input_image.width}x{input_image.height}
    """
    return result, param_info


with gr.Blocks() as demo:
    gr.Markdown("# Flux1-Kontext-Dev Pipeline Gradio Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="change her face to turn to the front side and the swimsuit color into dark blue bikini",
            )
            input_image = gr.Image(
                label="Input Image", value="default.jpg", type="pil", height=500
            )
            guidance = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=10.0, value=2.5, step=0.1
            )
            num_inference_steps = gr.Slider(
                label="Num Inference Steps", minimum=1, maximum=50, value=30, step=1
            )
            run_btn = gr.Button("Run")
        with gr.Column():
            output_img = gr.Image(label="Output Image", height=500)
            param_info_md = gr.Markdown(label="Parameter Info")

    def run_model(prompt, input_image, guidance, num_inference_steps):
        output_img, param_info = flux1_kontext_dev(
            prompt, input_image, guidance, num_inference_steps
        )
        return output_img, param_info

    run_btn.click(
        run_model,
        inputs=[prompt, input_image, guidance, num_inference_steps],
        outputs=[output_img, param_info_md],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
