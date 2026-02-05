import os
import torch
from diffusers import FluxKontextPipeline
from datetime import datetime
from PIL import Image
import gc
import atexit
import signal
import sys
import platform
import subprocess
import gradio as gr


def print_hardware_info():
    """Print hardware information at startup."""
    print("=" * 60)
    print("HARDWARE INFORMATION")
    print("=" * 60)

    # System info
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")

    # CPU info (Windows-specific)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                [
                    "wmic",
                    "cpu",
                    "get",
                    "name,numberofcores,numberoflogicalprocessors",
                    "/format:list",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().split("\n"):
                if line.strip() and "=" in line:
                    print(f"CPU {line.strip()}")
        except Exception:
            pass

        # RAM info
        try:
            result = subprocess.run(
                ["wmic", "memorychip", "get", "capacity", "/format:list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            total_ram = 0
            for line in result.stdout.strip().split("\n"):
                if line.strip().startswith("Capacity="):
                    total_ram += int(line.split("=")[1])
            if total_ram > 0:
                print(f"RAM Total: {total_ram / (1024**3):.1f} GB")
        except Exception:
            pass

    # GPU info via PyTorch
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"GPU {i} Memory: {props.total_memory / (1024**3):.1f} GB")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon): Available")
    else:
        print("GPU: Not available (CPU only)")

    print("=" * 60)


print_hardware_info()


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
if device_type == "cuda" or device_type == "mps":
    print(f"{device_type.upper()} detected, using bfloat16 for efficiency.")
    data_type = torch.bfloat16
elif device_type == "cpu":
        print("CPU detected, using float32 for better compatibility.")
        data_type = torch.float32
else:
    print("Unknown device, defaulting to float32.")
    data_type = torch.float32

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
        positions = [
            (0, 0),
            (tile_size[0], 0),
            (0, tile_size[1]),
            (tile_size[0], tile_size[1]),
        ]
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

    # Use width/height from sliders
    output_width = int(width)
    output_height = int(height)

    # Create a composite grid image
    # Use smaller tiles so the composite fits within reasonable dimensions
    tile_w = min(512, output_width)
    tile_h = min(512, output_height)
    composite_image = create_image_grid(input_images, tile_size=(tile_w, tile_h))

    composite_w, composite_h = composite_image.size
    # Ensure dimensions are divisible by 64
    composite_w = (composite_w // 64) * 64
    composite_h = (composite_h // 64) * 64
    composite_image = composite_image.resize(
        (composite_w, composite_h), Image.Resampling.LANCZOS
    )

    print(
        f"Created composite image grid: {composite_w}x{composite_h} from {len(input_images)} images"
    )
    print(f"Output dimensions: {output_width}x{output_height}")
    print(f"Prompt: {prompt}")

    generator = torch.Generator(device=device_type).manual_seed(int(seed))

    image = pipe(
        image=composite_image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=output_width,
        height=output_height,
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
        f"{script_name}_{timestamp}_w{output_width}_h{output_height}_guidance{guidance_scale}_steps{int(num_inference_steps)}_seed{int(seed)}_seqlen{int(max_sequence_length)}.png",
    )
    image.save(output_path)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return image, f"Image saved: {output_path}"


def get_output_dimensions(image):
    """Get output dimensions from first image (rounded to nearest 64)."""
    if image is None:
        return gr.update(), gr.update()
    w, h = image.size
    output_w = max(256, min(1536, round(w / 64) * 64))
    output_h = max(256, min(1536, round(h / 64) * 64))
    return output_w, output_h


# Default image paths
default_image_1 = "default01.png"
default_image_1_path = os.path.join(script_dir, default_image_1)

# Get initial output dimensions from default image
try:
    with Image.open(default_image_1_path) as img:
        w, h = img.size
        default_width = max(256, min(1536, round(w / 64) * 64))
        default_height = max(256, min(1536, round(h / 64) * 64))
except Exception:
    default_width, default_height = 512, 1024

with gr.Blocks(title="Flux.1 Kontext Dev Multi-Image Composition") as demo:
    gr.Markdown("# Flux.1 Kontext Dev Multi-Image Composition")
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
                    label="Image 1 (Required)",
                    type="pil",
                    height=280,
                    value=default_image_1_path,
                )
                input_image_2 = gr.Image(
                    label="Image 2 (Required)",
                    type="pil",
                    height=280,
                    value="beachsuncap.jpg",
                )
            with gr.Row():
                input_image_3 = gr.Image(
                    label="Image 3 (Optional)",
                    type="pil",
                    height=280,
                    value="sunglasses.jpg",
                )
                input_image_4 = gr.Image(
                    label="Image 4 (Optional)",
                    type="pil",
                    height=280,
                    value="bikini.jpg",
                )
            with gr.Row():
                width = gr.Slider(
                    256,
                    1536,
                    value=default_width,
                    step=64,
                    label="Output Width",
                    info="Output image width (initialized from Image 1)",
                )
                height = gr.Slider(
                    256,
                    1536,
                    value=default_height,
                    step=64,
                    label="Output Height",
                    info="Output image height (initialized from Image 1)",
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

    # Update width/height sliders when first image changes
    input_image_1.change(
        fn=get_output_dimensions,
        inputs=[input_image_1],
        outputs=[width, height],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
