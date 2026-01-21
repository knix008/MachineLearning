from transformers import logging

logging.set_verbosity_error()

import torch
import gradio as gr
from diffusers import FluxImg2ImgPipeline
from datetime import datetime
from PIL import Image
import os

DEFAULT_IMAGE_PATH = "default.jpg"

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

# Auto-detect device (Apple Silicon MPS or CPU)
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
    print("🍎 Using Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print("🎮 Using CUDA GPU")
else:
    device = "cpu"
    dtype = torch.float32  # CPU doesn't support bfloat16 well
    print("💻 Using CPU (Intel/AMD)")

print(f"Device: {device}, dtype: {dtype}")

# Load image-to-image pipeline (FLUX.1-dev supports img2img, FLUX.2-klein does not)
print("Loading FLUX.1-dev pipeline...")
pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
)

# Load FLUX.2-Turbo LoRA for turbo mode
print("Loading FLUX.2-Turbo LoRA...")
try:
    pipe.load_lora_weights(
        "fal/FLUX.2-dev-Turbo",
        weight_name="flux.2-turbo-lora.safetensors",
        adapter_name="turbo"
    )
    pipe.disable_adapters()  # Start with LoRA disabled, enable only for 8 steps
    print("✓ Turbo LoRA loaded successfully (will activate for 8-step inference)")
except Exception as e:
    print(f"⚠️ Warning: Could not load Turbo LoRA: {e}")
    print("Turbo mode will use custom sigmas only")

pipe = pipe.to(device)

# Memory optimization
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")


def generate_image(
    input_image, prompt, strength, guidance_scale, num_inference_steps, seed
):
    if input_image is None:
        input_image = Image.open(DEFAULT_IMAGE_PATH).convert("RGB")

    generator = torch.Generator(device=device).manual_seed(int(seed))

    # Enable Turbo mode for 8 steps
    if int(num_inference_steps) == 8:
        print("🚀 Turbo mode activated (8 steps)")
        # Enable LoRA for turbo inference
        pipe.set_adapters(["turbo"], adapter_weights=[1.0])

        output_image = pipe(
            prompt=prompt,
            image=input_image,
            width=input_image.width,
            height=input_image.height,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=8,
            sigmas=TURBO_SIGMAS,
            generator=generator,
        ).images[0]
    else:
        print(f"Standard mode ({num_inference_steps} steps)")
        # Disable LoRA for standard inference
        pipe.disable_adapters()

        output_image = pipe(
            prompt=prompt,
            image=input_image,
            width=input_image.width,
            height=input_image.height,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            generator=generator,
        ).images[0]

    # Get script filename without extension
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{script_name}_{timestamp}.png"
    output_image.save(filename)

    return output_image, f"이미지가 저장되었습니다: {filename}"


with gr.Blocks(title="FLUX.1 Image-to-Image with Turbo Mode") as demo:
    gr.Markdown("# FLUX.1 Image-to-Image Generator (with Turbo Mode)")
    gr.Markdown("""
    **🚀 Turbo Mode**: Set Inference Steps to **8** to automatically activate Turbo mode:
    - Uses FLUX.2-Turbo LoRA for optimized fast generation
    - Applies pre-shifted custom sigmas for better quality
    - Perfect for quick iterations and testing

    **Standard Mode**: Any other step value (e.g., 20, 28, 50) uses standard FLUX.1-dev inference without LoRA.
    """)
    gr.Markdown("**Note**: Actual inference steps = Inference Steps × Strength. For example, 20 steps with 0.85 strength = 17 actual steps.")
    gr.Markdown("""
    **Parameter Guide**:
    - **Strength** (0.0-1.0): Controls how much to change the input image. Higher values = more changes, lower values = preserve more of the original. Recommended: 0.65-0.85 for significant edits.
    - **Guidance Scale** (1.0-10.0): Controls how strictly the model follows your prompt. Higher values = stricter adherence to prompt (may cause artifacts), lower values = more creative freedom. Recommended: 3.5-7.0 for FLUX models.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", label="Input Image", value=DEFAULT_IMAGE_PATH, height=700
            )
            prompt = gr.Textbox(
                label="Prompt",
                value="Change the bikini color to dark blue, move her face to see the viewer, make the image 8k resolution quality, high detail, high quality, best quality, masterpiece, cinematic lighting.",
                lines=3,
            )

            with gr.Row():
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.85,
                    step=0.05,
                    label="Strength (how much to change the input image)",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale",
                )

            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=50, value=8, step=1, label="Inference Steps"
                )
                seed = gr.Number(value=42, label="Seed", precision=0)

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Output Image", height=700)
            status_text = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image,
            prompt,
            strength,
            guidance_scale,
            num_inference_steps,
            seed,
        ],
        outputs=[output_image, status_text],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
