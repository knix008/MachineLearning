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
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        # Add more parameters if needed
    )
    result[0].save(
        f"Flux-Kontext-Dev01_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    # result[0] is the output image
    return result[0]


with gr.Blocks() as demo:
    gr.Markdown("# Flux-Kontext-Dev Pipeline Gradio Demo")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="Change the car color to red, turn the headlights on",
            )
            input_image = gr.Image(label="Input Image", type="pil", width=500)
            guidance = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=10.0, value=2.5, step=0.1
            )
            num_inference_steps = gr.Slider(
                label="Num Inference Steps", minimum=1, maximum=100, value=30, step=1
            )
            run_btn = gr.Button("Run")
        with gr.Column():
            output_img = gr.Image(label="Output Image", width=500)

    def run_model(prompt, input_image, guidance, num_inference_steps):
        output_img = flux1_kontext_dev(
            prompt, input_image, guidance, num_inference_steps
        )
        return output_img

    run_btn.click(
        run_model,
        inputs=[prompt, input_image, guidance, num_inference_steps],
        outputs=output_img,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
