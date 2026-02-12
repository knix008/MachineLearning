import os

os.environ.pop("MallocStackLogging", None)
os.environ.pop("MallocStackLoggingDirectory", None)

import torch
from diffusers import FluxPipeline
from datetime import datetime
import gc
import atexit
import signal
import sys
import platform
import subprocess
import gradio as gr

prompt = "The image is a high-quality, photorealistic cosplay portrait of a young Korean woman with a soft, idol aesthetic. Physical Appearance: Face: She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera: She has long, straight jet-black hair with thick, straight-cut bangs (fringe) that frame her face. Attire (Blue & White Bunny Theme): Headwear: She wears tall, upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base, accented with a small white bow. Outfit: She wears a unique blue denim-textured bodysuit. It features a front zipper, silver buttons, and thin silver chains draped across the chest. The sides are constructed from semi-sheer white lace. Accessories: Around her neck is a blue bow tie attached to a white collar. She wears long, white floral lace fingerless sleeves that extend past her elbows, finished with blue cuffs and small black decorative ribbons. Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows. Pose: She is standing gracefully in front of the edge of a light-colored, vintage-style bed or cushioned bench. Her body is slightly angled toward the camera, creating a soft and inviting posture. Setting & Background: Location: A bright, high-key studio set designed to look like a clean, airy bedroom. Background: The background is dominated by large windows with white vertical blinds or curtains, allowing soft, diffused natural-looking light to flood the scene. The background is softly blurred (bokeh). Lighting: The lighting is bright, soft, and even, minimizing harsh shadows and giving the skin a glowing, porcelain appearance. Flux Prompt Prompt: A photorealistic, high-quality cosplay portrait of a beautiful Korean woman dressed in a blue and white bunny girl outfit. She has long straight black hair with hime-cut bangs and vibrant blue eyes. She wears tall blue bunny ears with white lace trim, a blue denim-textured bodysuit with a front zipper and white lace side panels, a blue bow tie, and long white lace sleeves. She is standing in front of a white bed in a bright, sun-drenched room with soft-focus white curtains. She is looking at the camera with a soft, innocent expression.8k resolution, high-key lighting, cinematic soft focus, detailed textures of denim and lace, gravure photography style. Key Stylistic Keywords Blue bunny girl, denim cosplay, white lace, high-key lighting, blue contact lenses, black hair with bangs, fishnet stockings, airy atmosphere, photorealistic, innocent and alluring, studio photography."


def detect_device():
    """Auto-detect the best available device and data type."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    else:
        return "cpu", torch.float32


def print_hardware_info(device, dtype):
    """Print hardware and software environment info."""
    print("=" * 60)
    print("Hardware Information")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")

    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            print(f"Chip: {chip}")
        except Exception:
            pass
        try:
            mem_bytes = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"], text=True
                ).strip()
            )
            print(f"System Memory: {mem_bytes / (1024 ** 3):.0f} GB")
        except Exception:
            pass

    if device == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {vram / (1024 ** 3):.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Data Type: {dtype}")

    if device == "mps":
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")

    print("=" * 60)


device_type, data_type = detect_device()
print_hardware_info(device_type, data_type)

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading model (T5-XXL only, CLIP disabled)...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    tokenizer=None,
    torch_dtype=data_type,
)

# Apply memory optimizations based on device
# NOTE: pipe.to(), enable_model_cpu_offload(), enable_sequential_cpu_offload()
#       are mutually exclusive — only use ONE of them.
if device_type == "cuda" or device_type == "cpu":
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    print("Memory optimization: model CPU offload, attention slicing (CUDA)")
elif device_type == "mps":
    pipe.to(device_type)
    print("Memory optimization: none (MPS)")
else:
    print("No device found.") 
    exit(1)

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
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Resources released!")


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle keyboard interrupt."""
    print("\nKeyboard interrupt received...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def generate_image(
    prompt,
    width,
    height,
    guidance_scale,
    num_inference_steps,
    seed,
    max_sequence_length,
):
    if not prompt or prompt.strip() == "":
        return None, "Please enter a prompt."

    print(f"Generating image with prompt: {prompt}")

    # Use CPU generator for MPS/CPU, device generator for CUDA
    gen_device = device_type if device_type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    # Encode prompt using T5 only (CLIP is disabled)
    encode_device = device_type if device_type != "cuda" else "cpu"
    text_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=int(max_sequence_length),
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder_2(
            text_inputs["input_ids"].to(encode_device),
            output_hidden_states=False,
        )[0]
    prompt_embeds = prompt_embeds.to(dtype=data_type)

    # Zero pooled embeddings (normally from CLIP, not needed with T5-only)
    pooled_prompt_embeds = torch.zeros(
        1, 768, dtype=data_type, device=prompt_embeds.device
    )

    image = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        width=int(width),
        height=int(height),
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        generator=generator,
    ).images[0]

    # Save with program name, timestamp, and parameters
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(
        script_dir,
        f"{script_name}_{timestamp}_{device_type.upper()}_w{int(width)}_h{int(height)}_guidance{guidance_scale}_steps{int(num_inference_steps)}_seed{int(seed)}_seqlen{int(max_sequence_length)}.png",
    )
    print("Saving image to:", output_path)
    image.save(output_path)

    # Cleanup
    gc.collect()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return image, f"Image saved: {output_path}"


with gr.Blocks(title="Flux.1 Kontext Text-to-Image") as demo:
    gr.Markdown("# Flux.1 Kontext Text-to-Image")
    gr.Markdown(
        f"Enter a prompt to generate an image from text. Device: **{device_type.upper()}**"
    )

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                value=prompt,
                lines=3,
            )
            with gr.Row():
                width = gr.Slider(
                    256,
                    1536,
                    value=768,
                    step=64,
                    label="Width",
                    info="Output image width in pixels",
                )
                height = gr.Slider(
                    256,
                    1536,
                    value=1536,
                    step=64,
                    label="Height",
                    info="Output image height in pixels",
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    1.0,
                    10.0,
                    value=3.5,
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
                seed = gr.Slider(
                    0,
                    1000,
                    value=42,
                    step=1,
                    label="Seed",
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
            prompt,
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
    try:
        demo.launch(inbrowser=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    finally:
        cleanup()
        sys.exit(0)
