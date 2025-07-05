import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion (CPU or CUDA)
# Make sure you have a valid access token from HuggingFace if using 'CompVis/stable-diffusion-v1-4' or similar
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_auth_token=True,  # Set your HF token in environment or pass it here
).to("cuda" if torch.cuda.is_available() else "cpu")


def generate_fashion(prompt, guidance_scale, num_inference_steps):
    image = pipe(
        prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps
    ).images[0]
    return image


with gr.Blocks() as demo:
    gr.Markdown("# Fashion Design Generator (Stable Diffusion)")
    gr.Markdown(
        "Enter a description for your fashion design and generate an image using Stable Diffusion!"
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Fashion Design Prompt",
            value="A summer dress with floral pattern, pastel colors",
        )
    with gr.Row():
        guidance_scale = gr.Slider(4, 20, value=7.5, step=0.5, label="Guidance Scale")
        num_steps = gr.Slider(10, 50, value=25, step=1, label="Inference Steps")
    btn = gr.Button("Generate Fashion Design")
    output = gr.Image(label="Generated Fashion Design")

    btn.click(
        fn=generate_fashion, inputs=[prompt, guidance_scale, num_steps], outputs=output
    )

if __name__ == "__main__":
    demo.launch()
