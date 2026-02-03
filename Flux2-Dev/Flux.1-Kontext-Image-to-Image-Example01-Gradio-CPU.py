import torch
from diffusers import FluxKontextPipeline
from datetime import datetime
import os
import gc
import atexit
import signal
import sys
import gradio as gr

# Define device type and data type
device_type = "cpu"
data_type = torch.float32

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=data_type
)
pipe.to(device_type)

print("Applying optimizations...")
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
print("Model loaded!")


def cleanup():
    """Release all resources before exit."""
    global pipe
    print("Releasing resources...")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Resources released!")


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle keyboard interrupt."""
    print("\nKeyboard interrupt received...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def generate_image(
    input_image,
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    max_sequence_length,
):
    if input_image is None:
        return None, "Please upload an input image."

    print(f"Generating image with prompt: {prompt}")

    generator = torch.Generator(device=device_type).manual_seed(int(seed))

    image = pipe(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=int(width),
        height=int(height),
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        max_sequence_length=int(max_sequence_length),
        generator=generator,
    ).images[0]

    # Save with program name, timestamp, and parameters
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        script_dir,
        f"{script_name}_{timestamp}_w{int(width)}_h{int(height)}_guidance{guidance_scale}_steps{int(num_inference_steps)}_seed{int(seed)}_seqlen{int(max_sequence_length)}.png",
    )
    image.save(output_path)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return image, f"Image saved: {output_path}"


with gr.Blocks(title="Flux Kontext Image-to-Image") as demo:
    gr.Markdown("# Flux Kontext Image-to-Image")
    gr.Markdown("Upload an image and describe the changes you want to make.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image", type="pil", value="default.png", height=600
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the changes you want...",
                value="Add a beach background with palm trees on the sunset beach. cinematic lighting, 4k quality, high detail.",
                info="Describe the modifications you want to apply to the input image",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid in the image...",
                value="blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text",
                info="Describe what you don't want in the generated image",
            )

            with gr.Row():
                width = gr.Slider(
                    256,
                    1536,
                    value=512,
                    step=64,
                    label="Width",
                    info="Output image width in pixels",
                )
                height = gr.Slider(
                    256,
                    1536,
                    value=1024,
                    step=64,
                    label="Height",
                    info="Output image height in pixels",
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    1.0,
                    10.0,
                    value=2.5,
                    step=0.1,
                    label="Guidance Scale",
                    info="How closely to follow the prompt. Higher = stronger effect",
                )
                num_inference_steps = gr.Slider(
                    1,
                    50,
                    value=28,
                    step=1,
                    label="Inference Steps",
                    info="Number of denoising steps. More steps = better quality but slower",
                )

            with gr.Row():
                seed = gr.Number(
                    value=42,
                    label="Seed",
                    precision=0,
                    info="Random seed for reproducibility. Same seed = same result",
                )
                max_sequence_length = gr.Slider(
                    128,
                    512,
                    value=256,
                    step=64,
                    label="Sequence Length",
                    info="Max token length for text encoder. Higher = longer prompts supported",
                )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Output Image", type="pil", height=600)
            status = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            seed,
            max_sequence_length,
        ],
        outputs=[output_image, status],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
