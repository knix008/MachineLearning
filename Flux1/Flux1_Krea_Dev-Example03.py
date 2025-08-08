import torch
from diffusers import FluxPipeline
import datetime
import os
import warnings
import gradio as gr

# Disable all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Disable Hugging Face symlinks warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load the model
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)

pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU.
pipe.enable_sequential_cpu_offload() #save some VRAM by offloading the model to CPU sequentially.
pipe.enable_attention_slicing() #save some VRAM by slicing the attention layers.

def generate_image(prompt, input_image, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image using FLUX.1-Krea-dev model"""
    try:
        # Set seed for reproducibility
        if seed == -1:
            # Generate random seed
            import random
            actual_seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(actual_seed)
        else:
            actual_seed = int(seed)
            torch.manual_seed(actual_seed)
        
        # Adjust dimensions based on input image if provided
        if input_image is not None:
            # Get input image dimensions
            img_width, img_height = input_image.size
            aspect_ratio = img_width / img_height
            
            # Calculate new dimensions maintaining aspect ratio
            # Use the larger dimension from the sliders as the base
            target_size = max(int(height), int(width))
            
            if aspect_ratio > 1:  # Landscape
                final_width = target_size
                final_height = int(target_size / aspect_ratio)
            else:  # Portrait or square
                final_height = target_size
                final_width = int(target_size * aspect_ratio)
            
            # Ensure dimensions are multiples of 64 (required by FLUX)
            final_width = (final_width // 64) * 64
            final_height = (final_height // 64) * 64
            
            # Ensure minimum size
            final_width = max(final_width, 256)
            final_height = max(final_height, 256)
            
            status_msg = f"Using input image aspect ratio: {final_width}x{final_height} (original: {img_width}x{img_height})"
        else:
            final_width = int(width)
            final_height = int(height)
            status_msg = f"Using specified dimensions: {final_width}x{final_height}"
        
        # Generate image
        if input_image is not None:
            # Image-to-image generation
            image = pipe(
                prompt,
                image=input_image,
                height=final_height,
                width=final_width,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                generator=torch.Generator().manual_seed(actual_seed)
            ).images[0]
        else:
            # Text-to-image generation
            image = pipe(
                prompt,
                height=final_height,
                width=final_width,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                generator=torch.Generator().manual_seed(actual_seed)
            ).images[0]
        
        # Save image with timestamp
        filename = f"flux1-krea-dev-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)
        
        return image, f"Image saved as: {filename}\n{status_msg}\nSeed: {actual_seed}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="FLUX.1-Krea-dev Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 FLUX.1-Krea-dev Image Generator")
    gr.Markdown("""
    **FLUX.1-Krea-dev 모델을 사용한 고품질 이미지 생성기**

    📝 **Text-to-Image**: 텍스트 프롬프트만으로 이미지 생성  
    🖼️ **Image-to-Image**: 입력 이미지를 기반으로 새로운 이미지 생성 (자동으로 원본 비율 유지)
    
    💡 **팁**: 더 나은 결과를 위해 구체적이고 상세한 프롬프트를 사용하세요!
    """)
   
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your image description...",
                value="8k, high quality, realistic, high detail, cinematic lighting",
                lines=3
            )
            
            input_image = gr.Image(
                label="Input Image (Optional)",
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                value="default.jpg",
                height=500
            )
            
            with gr.Row():
                height_input = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Height",
                    info="높이: 생성할 이미지의 세로 크기입니다. 입력 이미지가 있을 경우 비율에 맞춰 자동 조정됩니다. (권장: 1024)"
                )
                width_input = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Width", 
                    info="너비: 생성할 이미지의 가로 크기입니다. 입력 이미지가 있을 경우 비율에 맞춰 자동 조정됩니다. (권장: 1024)"
                )
            
            with gr.Row():
                guidance_scale_input = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=4.5,
                    step=0.1,
                    label="Guidance Scale",
                    info="가이던스 스케일: 프롬프트에 대한 모델의 충실도를 조절합니다. 높은 값은 프롬프트를 더 정확히 따르지만 창의성이 떨어질 수 있습니다. (권장: 3.5-7.5)"
                )
                num_inference_steps_input = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="Inference Steps",
                    info="추론 단계: 이미지 생성을 위한 디노이징 단계 수입니다. 높은 값은 더 정교한 이미지를 생성하지만 처리 시간이 길어집니다. (권장: 20-50)"
                )
            
            seed_input = gr.Number(
                label="Seed (-1 for random)",
                value=-1,
                precision=0,
                info="시드 값: 이미지 생성의 랜덤성을 제어합니다. 같은 시드 값을 사용하면 동일한 이미지를 재생성할 수 있습니다. -1을 입력하면 랜덤 시드를 사용합니다."
            )
            
            generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=500
            )
            output_text = gr.Textbox(
                label="Status",
                interactive=False,
                info="생성 상태 및 결과 정보가 표시됩니다. 사용된 시드 값도 함께 확인할 수 있습니다."
            )
    
    # Set up the generate button click event
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            input_image,
            height_input,
            width_input,
            guidance_scale_input,
            num_inference_steps_input,
            seed_input
        ],
        outputs=[output_image, output_text]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(inbrowser=True)