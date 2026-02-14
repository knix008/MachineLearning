import torch
from diffusers import FluxKontextPipeline
from datetime import datetime
import os
import gc
import time
import atexit
import signal
import sys
import platform
import gradio as gr
from PIL import Image

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

DEFAULT_IMAGE = "Test03.png"


# Pre-compute default image dimensions for slider initial values
def _get_default_dims(image_path):
    """Read default image and compute 64-aligned dimensions."""
    try:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        w = max(256, min(1536, round(orig_w / 64) * 64))
        h = max(256, min(1536, round(orig_h / 64) * 64))
        return orig_w, orig_h, w, h
    except Exception:
        return 512, 512, 512, 512


_orig_w, _orig_h, _default_w, _default_h = _get_default_dims(DEFAULT_IMAGE)


DEFAULT_PROMPT = "Show her back. Cinematic lighting, 4k, ultra-detailed texture, with perfect anatomy, perfect arms and legs structure, fashion vibe."

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text"
)


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

print("Loading model (CLIP + T5-XXL)...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=data_type,
)
pipe.to(device_type)

if device_type == "cuda":
    print("Applying optimizations for CUDA...")

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
elif device_type == "mps":
    print("MPS device detected, applying no optimizations...")
else:
    print("Applying CPU optimizations...")
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    print("Applying CPU optimizations...")

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
        return gr.update(), gr.update(), "No image loaded"
    orig_w, orig_h = image.size
    # Round to nearest 64 and clamp to slider range
    w = max(256, min(1536, round(orig_w / 64) * 64))
    h = max(256, min(1536, round(orig_h / 64) * 64))
    info = f"Original: {orig_w} x {orig_h} → Output: {w} x {h}"
    return w, h, info


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
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None:
        return None, "Please upload an input image."

    steps = int(num_inference_steps)
    start_time = time.time()

    progress(0.0, desc="Preparing...")
    print(f"Generating image: steps={steps}, guidance={guidance_scale}, seed={int(seed)}")

    # Use "cpu" for generator on MPS as it's more stable
    generator_device = "cpu" if device_type == "mps" else device_type
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    def step_callback(_pipe, step_index, _timestep, callback_kwargs):
        current = step_index + 1
        elapsed = time.time() - start_time
        ratio = current / steps
        progress_val = 0.05 + ratio * 0.85
        msg = f"Step {current}/{steps} ({elapsed:.1f}s elapsed)"
        progress(progress_val, desc=msg)
        print(f"  {msg}")
        return callback_kwargs

    progress(0.05, desc="Starting inference...")

    image = pipe(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=int(width),
        height=int(height),
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        max_sequence_length=int(max_sequence_length),
        generator=generator,
        callback_on_step_end=step_callback,
    ).images[0]

    progress(0.95, desc="Saving image...")

    # Save with program name, timestamp, and parameters
    elapsed = time.time() - start_time
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        script_dir,
        f"{script_name}_{timestamp}_{device_type}_guidance{guidance_scale}_steps{steps}_seed{int(seed)}_seqlen{int(max_sequence_length)}.png",
    )
    print(f"Saving image to: {output_path}")
    image.save(output_path)

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    progress(1.0, desc="Done!")
    print(f"Done! Total time: {elapsed:.1f}s")

    return image, f"Done ({elapsed:.1f}s) | Saved: {output_path}"


with gr.Blocks(title="Flux.1 Kontext Image-to-Image 테스트") as demo:
    gr.Markdown("# Flux.1 Kontext Image-to-Image 테스트")
    gr.Markdown("Upload an image and describe the changes you want to make.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image", type="pil", value=DEFAULT_IMAGE, height=600
            )
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the changes you want...",
                value=DEFAULT_PROMPT,
                info="Describe the modifications you want to apply to the input image",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="What to avoid in the image...",
                value=DEFAULT_NEGATIVE_PROMPT,
                info="Describe what you don't want in the generated image",
            )

            dimension_info = gr.Textbox(
                label="Image Dimensions",
                interactive=False,
                value=f"Original: {_orig_w} x {_orig_h} → Output: {_default_w} x {_default_h}",
            )

            with gr.Row():
                width = gr.Slider(
                    256,
                    1536,
                    value=_default_w,
                    step=64,
                    label="Width",
                    info="Output image width in pixels (auto-set from input, adjustable)",
                )
                height = gr.Slider(
                    256,
                    1536,
                    value=_default_h,
                    step=64,
                    label="Height",
                    info="Output image height in pixels (auto-set from input, adjustable)",
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
                    info="Max token length for text encoder. Higher = longer prompts supported but more VRAM",
                )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Output Image", type="pil", height=1000)
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

    # Update width/height sliders and dimension info when input image changes
    input_image.change(
        fn=get_image_dimensions,
        inputs=[input_image],
        outputs=[width, height, dimension_info],
    )


if __name__ == "__main__":
    try:
        demo.launch(inbrowser=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    finally:
        cleanup()
        sys.exit(0)
