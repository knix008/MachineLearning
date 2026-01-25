import torch
from diffusers import Flux2Pipeline
from datetime import datetime
import os
import gradio as gr

# 일반 FLUX.2-dev 모델 사용 (양자화 모델에 문제가 있음)
repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device.upper()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

pipe = Flux2Pipeline.from_pretrained(
    repo_id, 
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,  # CPU 메모리 사용량 감소
)

if torch.cuda.is_available():
    # CPU offload를 사용하여 GPU 메모리 부담 줄이기
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()  # 순차적 CPU offload로 메모리 최적화
    pipe.vae.enable_tiling()  # VAE 타일링으로 메모리 사용량 감소
    pipe.vae.enable_slicing()  # VAE 슬라이싱 활성화
else:
    pipe = pipe.to(device)
print("모델 로딩 완료!")

def generate_image(prompt, num_inference_steps, guidance_scale, seed, height, width):
    """Generate image from text prompt"""
    try:
        # Set random seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        image = pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]
        
        # Save image with timestamp
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{script_name}_{timestamp}.png"
        image.save(output_filename)
        
        return image, f"Image saved as: {output_filename} (Size: {width}x{height})"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="FLUX.2-Dev Image Generator") as demo:
    gr.Markdown("# FLUX.2-Dev Image Generator")
    gr.Markdown("Generate high-quality images from text prompts using FLUX.2-Dev model")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your image description here...",
                lines=5,
                value="Highly realistic, 4k, high-quality, high resolution, beautiful korean woman model photography. having black medium-length hair reaching her shoulders, tied back, wearing a red bikini, looking at the viewer. Perfect anatomy, solid orange backdrop, using a camera setup that mimics a large aperture f/1.4, ar 9:16, style raw.s"
            )
            
            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=28,
                    step=1,
                    label="Inference Steps (Higher = Better Quality, Slower) : 28 recommended"
                )
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    label="Guidance Scale (3-7 recommended)"
                )
            
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=512,
                    step=64,
                    label="Width (Multiples of 64 recommended)"
                )
                height_slider = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Height"
                )
            
            seed_input = gr.Number(
                label="Seed (for reproducibility)",
                value=42,
                precision=0
            )
            
            generate_btn = gr.Button("Generate Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
            output_status = gr.Textbox(label="Status", lines=2)
    
    # Button click event
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, steps_slider, guidance_slider, seed_input, height_slider, width_slider],
        outputs=[output_image, output_status]
    )
    
    gr.Markdown("---")
    gr.Markdown("### Tips:")
    gr.Markdown("- **Inference Steps**: 10-28 steps provide a good balance between quality and speed")
    gr.Markdown("- **Guidance Scale**: Higher values follow the prompt more closely (3-7 recommended)")
    gr.Markdown("- **Width/Height**: Use multiples of 64. Larger sizes require more VRAM (1024x1024 recommended)")
    gr.Markdown("- **Seed**: Use the same seed to reproduce the same image")

# Launch the app
if __name__ == "__main__":
    demo.launch(inbrowser=True)