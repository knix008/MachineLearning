import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os
import gradio as gr

device = "cuda"
dtype = torch.bfloat16

print("모델 로딩 중... 기다려주세요.")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype
)
pipe = pipe.to(device)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")


def generate_image(prompt, height, width, guidance_scale, num_steps, seed):
    """Generate image from text prompt"""
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_steps),
        generator=generator,
    ).images[0]
    
    # Save image with timestamp
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{script_name}_{timestamp}.png"
    image.save(filename)
    print(f"이미지가 저장되었습니다: {filename}")
    
    return image


# Default prompt
default_prompt = "Highly realistic, 4k, high-quality, high resolution,  beautiful instagram-style skinny korean girl full body photography with perfect anatomy, facing her body to the right side. She is looking at the viewer. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw."

# Create Gradio interface
with gr.Blocks(title="FLUX.2 Klein 9B Image Generator") as demo:
    gr.Markdown("# FLUX.2 Klein 9B Text-to-Image Generator")
    gr.Markdown("Generate high-quality images using FLUX.2 Klein 9B model")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="📝 Prompt",
                placeholder="Enter your prompt here...",
                value=default_prompt,
                lines=5,
                info="Describe the image you want to generate"
            )
            
            gr.Markdown("### 🖼️ Image Dimensions")
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Width (pixels)",
                    info="Image width in pixels"
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Height (pixels)",
                    info="Image height in pixels"
                )
            
            gr.Markdown("### ⚙️ Generation Settings")
            with gr.Row():
                guidance_slider = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1,
                    label="Guidance Scale",
                    info="How closely to follow the prompt (higher = more strict)"
                )
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=4,
                    step=1,
                    label="Inference Steps",
                    info="Number of denoising steps (more steps = higher quality but slower)"
                )
            
            seed_input = gr.Number(
                label="🎲 Seed",
                value=42,
                precision=0,
                info="Random seed for reproducibility (same seed = same result)"
            )
            
            generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="🖼️ Generated Image", type="pil", height=800)
            gr.Markdown("_Images are automatically saved with timestamp_")
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, height_slider, width_slider, guidance_slider, steps_slider, seed_input],
        outputs=output_image
    )

# Launch the interface
demo.launch(share=False, inbrowser=True)
