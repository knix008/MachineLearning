import os
os.environ["MallocStackLogging"] = "0"

import torch
from diffusers import FluxKontextPipeline
from datetime import datetime
from PIL import Image
import gc
import atexit
import signal
import sys
import gradio as gr

# Define device type and data type
device_type = "mps"
data_type = torch.bfloat16

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=data_type
)
pipe.to(device_type)
print("Model loaded!")


def cleanup():
    """Release all resources before exit."""
    global pipe
    print("Releasing resources...")
    try:
        del pipe
    except NameError:
        pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Resources released!")


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle keyboard interrupt."""
    print("\nKeyboard interrupt received...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_image_dimensions(image):
    """Get image dimensions and round to nearest step of 64."""
    if image is None:
        return gr.update(), gr.update()
    w, h = image.size
    # Round to nearest 64 and clamp to slider range
    w = max(256, min(1536, round(w / 64) * 64))
    h = max(256, min(1536, round(h / 64) * 64))
    return w, h


def generate_image(
    input_image_1,
    input_image_2,
    input_image_3,
    input_image_4,
    prompt,
    negative_prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    max_sequence_length,
):
    # Collect all non-None images and resize to target dimensions
    target_width = int(width)
    target_height = int(height)
    input_images = []
    for img in [input_image_1, input_image_2, input_image_3, input_image_4]:
        if img is not None:
            # Resize image to target dimensions
            resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            input_images.append(resized_img)

    if len(input_images) == 0:
        return None, "Please upload at least one image."

    if len(input_images) < 2:
        return None, "Please upload at least 2 images for composition."

    print(f"Generating image with {len(input_images)} input images")
    print(f"All images resized to {target_width}x{target_height}")
    print(f"Prompt: {prompt}")

    generator = torch.Generator(device=device_type).manual_seed(int(seed))

    image = pipe(
        image=input_images,
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
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return image, f"Image saved: {output_path}"


with gr.Blocks(title="Flux Kontext Multi-Image Composition") as demo:
    gr.Markdown("# Flux Kontext Multi-Image Composition")
    gr.Markdown(
        "Upload multiple images and describe how to combine them. "
        "Use 'image 1', 'image 2', etc. in your prompt to reference each image."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Images (at least 2 required)")
            with gr.Row():
                input_image_1 = gr.Image(
                    label="Image 1 (Required)", type="pil", height=280
                )
                input_image_2 = gr.Image(
                    label="Image 2 (Required)", type="pil", height=280
                )
            with gr.Row():
                input_image_3 = gr.Image(
                    label="Image 3 (Optional)", type="pil", height=280
                )
                input_image_4 = gr.Image(
                    label="Image 4 (Optional)", type="pil", height=280
                )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe how to combine the images...",
                value="Combine the person from image 1 with the other images. Keep the person's pose and clothing. cinematic lighting, 4k quality, high detail.",
                info="Use 'image 1', 'image 2', etc. to reference each uploaded image",
                lines=3,
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
            input_image_1,
            input_image_2,
            input_image_3,
            input_image_4,
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

    # Update width/height sliders when first input image changes
    input_image_1.change(
        fn=get_image_dimensions,
        inputs=[input_image_1],
        outputs=[width, height],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
