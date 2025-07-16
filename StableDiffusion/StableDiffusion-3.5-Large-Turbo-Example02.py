from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel
import torch
import gradio as gr
import os
from datetime import datetime

# Global variables for model initialization
model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

print("Initializing Stable Diffusion 3.5 Large Turbo model...")
try: 
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        text_encoder_3=t5_nf4,
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()
    print("Model initialized successfully!")
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    exit(1)

def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale, max_sequence_length, width, height, seed):
    """Generate image using Stable Diffusion 3.5 Large Turbo"""
    try:
        # Set seed for reproducibility
        if seed != -1:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            width=width,
            height=height,
            generator=generator
        )
        
        image = result.images[0]
        
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sd35_turbo_{timestamp}.png"
        image.save(filename)
        
        return image, f"Image saved as: {filename}"
        
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Stable Diffusion 3.5 Large Turbo Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Stable Diffusion 3.5 Large Turbo Image Generator")
        gr.Markdown("Generate high-quality images using Stable Diffusion 3.5 Large Turbo with customizable parameters")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                    value="a beautiful skinny woman wearing a high legged red bikini, walking on the sunny beach, photorealistic, 8k resolution, ultra detailed, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality, girl, solo, full body, looking at viewer, long hair, blue eyes, smiling, good fingers, good hands, good face, perfect anatomy"
                )
                
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want in the image...",
                    lines=2,
                    value="bad anatomy, text, watermark, logo, signature, low quality, blurry, bad quality, low resolution, cropped image, bad fingers, bad hands, bad face, ugly, worst quality, low quality, normal quality, jpeg artifacts, error, missing fingers, extra digit, fewer digits, long neck, long body, long arms, long legs, long fingers, long toes, long hair, bad lighting, bad shadows"
                )
                
                with gr.Row():
                    num_inference_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps"
                    )
                    
                    guidance_scale = gr.Slider(
                        minimum=0.0,
                        maximum=20.0,
                        value=0.0,
                        step=0.1,
                        label="Guidance Scale"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=64,
                        label="Width"
                    )
                    
                    height = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=64,
                        label="Height"
                    )
                
                with gr.Row():
                    max_sequence_length = gr.Slider(
                        minimum=128,
                        maximum=1024,
                        value=512,
                        step=64,
                        label="Max Sequence Length"
                    )
                    
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )
                
                generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(label="Generated Image", type="pil")
                output_info = gr.Textbox(label="Generation Info", lines=2)
        
        # Event handlers
        generate_btn.click(
            fn=generate_image,
            inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, max_sequence_length, width, height, seed],
            outputs=[output_image, output_info]
        )
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=False, inbrowser=True
    )