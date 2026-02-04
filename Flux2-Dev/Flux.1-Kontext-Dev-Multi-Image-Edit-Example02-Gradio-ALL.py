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
def get_device():
    """Detect available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device_type = get_device()
data_type = torch.bfloat16 if device_type != "cpu" else torch.float32

DEFAULT_PROMPT = "Make her wearing the beach sun cap, the sunglasses and the bikini. Keep the pose. cinematic lighting, 4k quality, high detail."

script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Loading model on {device_type.upper()} (dtype: {data_type})...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=data_type
)

# Memory optimization settings
if device_type == "cuda" or device_type == "cpu":
    # CUDA-specific optimizations
    pipe.to(device_type)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    print("Enabled: model_cpu_offload")
else:
    # MPS-specific optimizations
    print("MPS device detected, no memory optimization!!!")


print(f"Model loaded on {device_type.upper()}!")


def cleanup():
    """Release all resources before exit."""
    global pipe
    print("Releasing resources...")
    try:
        del pipe
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
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


def create_image_grid(images, tile_size=(512, 512)):
    """Create a grid of images for multi-image conditioning.

    For 2 images: side by side (1x2)
    For 3-4 images: 2x2 grid
    """
    if len(images) == 0:
        return None

    # Resize all images to tile size
    resized = [img.resize(tile_size, Image.Resampling.LANCZOS) for img in images]

    if len(resized) == 1:
        return resized[0]
    elif len(resized) == 2:
        # Side by side
        grid_width = tile_size[0] * 2
        grid_height = tile_size[1]
        grid = Image.new("RGB", (grid_width, grid_height))
        grid.paste(resized[0], (0, 0))
        grid.paste(resized[1], (tile_size[0], 0))
        return grid
    else:
        # 2x2 grid (fill empty slots with black if needed)
        grid_width = tile_size[0] * 2
        grid_height = tile_size[1] * 2
        grid = Image.new("RGB", (grid_width, grid_height), (0, 0, 0))
        positions = [(0, 0), (tile_size[0], 0), (0, tile_size[1]), (tile_size[0], tile_size[1])]
        for i, img in enumerate(resized[:4]):
            grid.paste(img, positions[i])
        return grid


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

    # Create a composite grid image
    # Use smaller tiles so the composite fits within reasonable dimensions
    tile_w = min(512, int(width))
    tile_h = min(512, int(height))
    composite_image = create_image_grid(input_images, tile_size=(tile_w, tile_h))

    composite_w, composite_h = composite_image.size
    # Ensure dimensions are divisible by 64
    composite_w = (composite_w // 64) * 64
    composite_h = (composite_h // 64) * 64
    composite_image = composite_image.resize((composite_w, composite_h), Image.Resampling.LANCZOS)

    print(f"Created composite image grid: {composite_w}x{composite_h} from {len(input_images)} images")
    print(f"Prompt: {prompt}")

    generator = torch.Generator(device=device_type).manual_seed(int(seed))

    image = pipe(
        image=composite_image,
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
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return image, f"Image saved: {output_path}"


# Get default image dimensions for slider initial values
default_image_1 = "default01.png"
default_image_1_path = os.path.join(script_dir, default_image_1)
try:
    with Image.open(default_image_1_path) as img:
        default_w, default_h = img.size
        default_w = max(256, min(1536, round(default_w / 64) * 64))
        default_h = max(256, min(1536, round(default_h / 64) * 64))
except Exception:
    default_w, default_h = 512, 1024

with gr.Blocks(title="Flux.1 Kontext Multi-Image Composition") as demo:
    gr.Markdown("# Flux.1 Kontext Multi-Image Composition")
    gr.Markdown(
        "Upload multiple images and describe how to combine them. "
        "For 2 images: use 'left image' and 'right image'. "
        "For 3-4 images: use 'top-left', 'top-right', 'bottom-left', 'bottom-right'."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Images (at least 2 required)")
            with gr.Row():
                input_image_1 = gr.Image(
                    label="Image 1 (Required)", type="pil", height=280, value=default_image_1_path
                )
                input_image_2 = gr.Image(
                    label="Image 2 (Required)", type="pil", height=280, value="beachsuncap.jpg"
                )
            with gr.Row():
                input_image_3 = gr.Image(
                    label="Image 3 (Optional)", type="pil", height=280, value="sunglasses.jpg"
                )
                input_image_4 = gr.Image(
                    label="Image 4 (Optional)", type="pil", height=280, value="bikini.jpg"
                )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe how to combine the images...",
                value=DEFAULT_PROMPT,
                info="Use 'left/right' for 2 images, or 'top-left/top-right/bottom-left/bottom-right' for 3-4 images",
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
                    value=default_w,
                    step=64,
                    label="Width",
                    info="Output image width in pixels",
                )
                height = gr.Slider(
                    256,
                    1536,
                    value=default_h,
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
