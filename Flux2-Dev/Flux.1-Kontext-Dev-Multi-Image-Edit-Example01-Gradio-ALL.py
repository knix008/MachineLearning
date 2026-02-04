import torch
from diffusers import FluxKontextPipeline
from datetime import datetime
import os
import gc
import atexit
import signal
import sys
import platform
import gradio as gr

# Print platform information
print("=" * 50)
print("Platform Information")
print("=" * 50)
print(f"OS: {platform.system()} {platform.release()}")
print(f"OS Version: {platform.version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"Python Version: {platform.python_version()}")
print(f"PyTorch Version: {torch.__version__}")
print("=" * 50)

DEFAULT_IMAGE_1 = "default01.png"
DEFAULT_IMAGE_2 = None
DEFAULT_IMAGE_3 = None
DEFAULT_IMAGE_4 = None
DEFAULT_PROMPT = "Combine the person from image 1 with the other images. Keep the person's pose and clothing. cinematic lighting, 4k quality, high detail."
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text"

# Detect and set device type and data type
def get_device_and_dtype():
    """Detect the best available device and appropriate data type."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16  # Use float16 for better performance on CUDA
        print(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.bfloat16  # Use bfloat16 for MPS
        print("MPS (Apple Silicon) device detected")
    else:
        device = "cpu"
        dtype = torch.float32  # Use float32 for CPU (float16 not well supported)
        print("Using CPU device")
    return device, dtype


device_type, data_type = get_device_and_dtype()
print(f"Using device: {device_type}, dtype: {data_type}")

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading model...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=data_type
)
pipe.to(device_type)

if device_type == "cuda" or device_type == "cpu":
    print("Applying optimizations...")
    # CPU offload helps manage VRAM on CUDA devices
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
else:
    print("No memory optimizations applied...")

print("Model loaded!")


def cleanup():
    """Release all resources before exit."""
    global pipe
    print("Releasing resources...")
    if "pipe" in globals() and pipe is not None:
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
    # Collect all non-None images
    input_images = []
    for img in [input_image_1, input_image_2, input_image_3, input_image_4]:
        if img is not None:
            input_images.append(img)

    if len(input_images) == 0:
        return None, "Please upload at least one image."

    if len(input_images) < 2:
        return None, "Please upload at least 2 images for composition."

    print(f"Generating image with {len(input_images)} input images")
    print(f"Prompt: {prompt}")

    # Use "cpu" for generator on MPS as it's more stable
    generator_device = "cpu" if device_type == "mps" else device_type
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

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
        f"{script_name}_{timestamp}_guidance{guidance_scale}_steps{int(num_inference_steps)}_seed{int(seed)}_seqlen{int(max_sequence_length)}.png",
    )
    image.save(output_path)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return image, f"Image saved: {output_path}"


with gr.Blocks(title="Flux.1 Kontext Dev Multi-Image Composition") as demo:
    gr.Markdown("# Flux.1 Kontext Dev Multi-Image Composition")
    gr.Markdown(
        "Upload multiple images and describe how to combine them. "
        "Use 'image 1', 'image 2', etc. in your prompt to reference each image."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Images (at least 2 required)")
            with gr.Row():
                input_image_1 = gr.Image(
                    label="Image 1 (Required)", type="pil", value=DEFAULT_IMAGE_1, height=280
                )
                input_image_2 = gr.Image(
                    label="Image 2 (Required)", type="pil", value=DEFAULT_IMAGE_2, height=280
                )
            with gr.Row():
                input_image_3 = gr.Image(
                    label="Image 3 (Optional)", type="pil", value=DEFAULT_IMAGE_3, height=280
                )
                input_image_4 = gr.Image(
                    label="Image 4 (Optional)", type="pil", value=DEFAULT_IMAGE_4, height=280
                )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe how to combine the images...",
                value=DEFAULT_PROMPT,
                info="Use 'image 1', 'image 2', etc. to reference each uploaded image",
                lines=3,
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid in the image...",
                value=DEFAULT_NEGATIVE_PROMPT,
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

    # Also update dimensions on initial app load for default image
    demo.load(
        fn=get_image_dimensions,
        inputs=[input_image_1],
        outputs=[width, height],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
