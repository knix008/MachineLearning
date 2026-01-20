import torch
import gradio as gr
from diffusers import Flux2Pipeline
from datetime import datetime
import os

# https://www.aitimes.com/news/articleView.html?idxno=205183 to get the access right.

# from huggingface_hub import login
#
# access_token = "hf_"
# login(access_token)

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

print("모델 로딩 중...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)
print("모델 로딩 완료!")


def generate_image(prompt, guidance_scale=2.5, height=1024, width=1024):
    """Generate image from prompt using FLUX.2-dev Turbo model."""
    try:
        image = pipe(
            prompt=prompt,
            sigmas=TURBO_SIGMAS,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=8,
        ).images[0]
        
        # Generate filename with script name, date and time
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{script_name}_{timestamp}.png"
        
        image.save(output_filename)
        print(f"이미지가 저장되었습니다: {output_filename}")
        
        return image
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return None


# Create Gradio interface
with gr.Blocks(title="FLUX.2-dev Turbo Image Generator") as demo:
    gr.Markdown("# FLUX.2-dev Turbo 이미지 생성기")
    gr.Markdown("프롬프트를 입력하면 AI가 이미지를 생성합니다.")
    
    default_prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="프롬프트",
                placeholder="생성하고 싶은 이미지에 대해 설명해주세요...",
                value=default_prompt,
                lines=4
            )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.5,
                    value=2.5
                )
            
            with gr.Row():
                height = gr.Slider(
                    label="높이",
                    minimum=512,
                    maximum=1024,
                    step=256,
                    value=1024
                )
                width = gr.Slider(
                    label="너비",
                    minimum=512,
                    maximum=1024,
                    step=256,
                    value=1024
                )
            
            generate_btn = gr.Button("이미지 생성", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(
                label="생성된 이미지",
                type="pil"
            )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, guidance_scale, height, width],
        outputs=image_output
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation",
            "A serene landscape with mountains, lake, and sunset",
            "A modern, sleek living room with minimalist design"
        ],
        inputs=prompt_input
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
