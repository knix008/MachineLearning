import torch
from diffusers import FluxPipeline
from datetime import datetime
from PIL import Image
import os
import warnings
import gradio as gr

# Suppress the add_prefix_spade warning
warnings.filterwarnings("ignore", message=".*add_prefix_spade.*")

# Set device and data type
device = "cpu"
dtype = torch.float32

# Load text-to-image pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
).to(device)

# Enable memory optimizations
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.enable_attention_slicing(1)  # reduce memory usage further
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")

prompt_input = "A highly realistic, high-quality photo of a beautiful Instagram-style girl with black, medium-length hair tied back casually. Her hazel eyes sparkle with happiness as she smiles. She wears a red bikini with perfect anatomy and precise details. Her skin appears natural with visible pores, avoiding overly smooth or filtered looks."

# prompt_input = "A highly realistic, high-quality photo of a beautiful Instagram-style girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin should appear natural, with visible pores. The overall atmosphere is bright and joyful, reflecting the sunny."

def generate_image(prompt, width, height, guidance_scale, num_inference_steps, seed):
    """
    Generate an image based on the provided parameters
    """
    try:
        # Run the pipeline
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}.png"
        image.save(filename)
        
        return image, f"✓ 이미지가 저장되었습니다: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Flux.1-dev Image Generator") as interface:
    gr.Markdown("# 🎨 Flux.1-dev Image Generator")
    gr.Markdown("AI를 사용하여 텍스트에서 이미지를 생성합니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input parameters
            prompt = gr.Textbox(
                label="프롬프트",
                value=prompt_input,
                lines=3,
                placeholder="이미지에 대한 설명을 입력하세요 (77단어 이하 권장)"
            )
            
            with gr.Row():
                width = gr.Slider(
                    label="이미지 너비",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=768
                )
                height = gr.Slider(
                    label="이미지 높이",
                    minimum=256,
                    maximum=1024,
                    step=64,
                    value=1024
                )
            
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale (프롬프트 강도)",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=4.0
                )
                num_inference_steps = gr.Slider(
                    label="추론 스텝",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=20
                )
            
            seed = gr.Number(
                label="시드 (일관된 결과를 위해)",
                value=400,
                precision=0
            )
            
            generate_btn = gr.Button("🚀 이미지 생성", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(label="생성된 이미지")
            output_message = gr.Textbox(label="상태", interactive=False)
    
    # Connect the generate button to the function
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, width, height, guidance_scale, num_inference_steps, seed],
        outputs=[output_image, output_message]
    )
    
    gr.Markdown("---")
    gr.Markdown("""
    ### 팁:
    - **프롬프트**: 자세할수록 좋습니다. 예: "여자, 미소, 해변, 빨간 비키니"
    - **Guidance Scale**: 낮을수록 창의적, 높을수록 프롬프트에 정확합니다 (권장: 4-15)
    - **추론 스텝**: 높을수록 품질이 좋지만 시간이 더 걸립니다 (권장: 20-28)
    - **시드**: 같은 시드를 사용하면 같은 결과를 얻습니다
    """)

# Launch the interface
if __name__ == "__main__":
    interface.launch(inbrowser=True)
