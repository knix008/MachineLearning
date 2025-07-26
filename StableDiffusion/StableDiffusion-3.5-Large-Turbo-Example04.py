from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel
import torch
import gradio as gr
import os
from datetime import datetime
from PIL import Image

# Global variables for model initialization
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

print("Initializing Stable Diffusion 3.5 Large Turbo model...")
try:
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )

    t5_nf4 = T5EncoderModel.from_pretrained(
        "diffusers/t5-nf4", torch_dtype=torch.bfloat16
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16,
    )

    pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
    pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
    pipe.enable_attention_slicing(1)  # Slice attention computation
    pipe.enable_vae_slicing()  # Slice VAE computation
    print("Model initialized successfully!")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    exit(1)


def resize_to_multiple_of_16(image):
    """Resize PIL image to closest size (down) that is divisible by 16, keeping aspect ratio."""
    w, h = image.size
    # Calculate new size
    new_w = (w // 16) * 16
    new_h = (h // 16) * 16
    # Keep aspect ratio
    aspect = w / h
    if new_w / aspect > new_h:
        new_h = int(new_w / aspect)
        new_h = (new_h // 16) * 16
    else:
        new_w = int(new_h * aspect)
        new_w = (new_w // 16) * 16
    return image.resize((new_w, new_h), Image.LANCZOS)


def generate_image(
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    max_sequence_length,
    width,
    height,
    seed,
    input_image,
):
    """Generate image using Stable Diffusion 3.5 Large Turbo"""
    try:
        # Set seed for reproducibility
        if seed != -1:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        # Resize input image if provided
        if input_image is not None:
            input_image = resize_to_multiple_of_16(input_image)

        # Generate image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            width=width,
            height=height,
            generator=generator,
            image=input_image if input_image is not None else None,
        )
        image = result.images[0]
        return image
    except Exception as e:
        return None


# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="Stable Diffusion 3.5 Large Turbo Generator", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# 🎨 Stable Diffusion 3.5 Large Turbo Image Generator")
        gr.Markdown(
            "텍스트 프롬프트와 입력 이미지를 함께 사용하여 이미지를 생성할 수 있습니다."
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value="a beautiful skinny woman wearing a high legged red bikini, walking on the sunny beach, photorealistic, 8k resolution, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, girl, solo, full body, looking at viewer, long hair, blue eyes, smiling, good fingers, good hands, good face, perfect anatomy",
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want in the image...",
                    lines=2,
                    value="bad anatomy, text, watermark, logo, signature, low quality, blurry, bad quality, low resolution, cropped image, bad fingers, bad hands, bad face, ugly, worst quality, low quality, normal quality, jpeg artifacts, error, missing fingers, extra digit, fewer digits, long neck, long body, long arms, long legs, long fingers, long toes, long hair, bad lighting, bad shadows",
                )

                input_image = gr.Image(
                    label="Input Image (optional)",
                    type="pil",
                    tool="editor",
                )

                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=1, maximum=50, value=20, step=1, label="Inference Steps"
                    )

                    guidance_scale = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=0.0,
                        step=0.1,
                        label="Guidance Scale",
                    )

                with gr.Row():
                    width = gr.Slider(
                        minimum=512, maximum=1536, value=1024, step=64, label="Width"
                    )

                    height = gr.Slider(
                        minimum=512, maximum=1536, value=1024, step=64, label="Height"
                    )

                with gr.Row():
                    max_sequence_length = gr.Slider(
                        minimum=128,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Max Sequence Length",
                    )

                    seed = gr.Number(
                        label="Seed (-1 for random)", value=-1, precision=0
                    )

                generate_btn = gr.Button(
                    "🎨 Generate Image", variant="primary", size="lg"
                )

            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image", type="pil")

        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                num_inference_steps,
                guidance_scale,
                max_sequence_length,
                width,
                height,
                seed,
                input_image,  # 입력 이미지 추가
            ],
            outputs=[output_image],
        )
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(inbrowser=True)
