import torch
from diffusers import FluxPipeline
import datetime
import os
import warnings
import gradio as gr

# Disable all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Disable Hugging Face symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

model_id = "black-forest-labs/FLUX.1-Krea-dev"

# Load the model
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU.
pipe.enable_sequential_cpu_offload()  # save some VRAM by offloading the model to CPU sequentially.
pipe.enable_attention_slicing()  # save some VRAM by slicing the attention layers.
print("Model loaded successfully.")

def generate_image(
    prompt, negative_prompt, height, width, guidance_scale, num_inference_steps, seed
):
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

        final_width = int(width)
        final_height = int(height)
        status_msg = f"Using specified dimensions: {final_width}x{final_height}"

        # Text-to-image generation with negative prompt
        image = pipe(
            prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            height=final_height,
            width=final_width,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            generator=torch.Generator().manual_seed(actual_seed),
        ).images[0]

        # Save image with timestamp
        filename = f"flux1-krea-dev-example03-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        image.save(filename)

        return image, f"Image saved as: {filename}\n{status_msg}\nSeed: {actual_seed}"

    except Exception as e:
        return None, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="FLUX.1-Krea-dev Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 FLUX.1-Krea-dev Image Generator")
    gr.Markdown(
        """
    **FLUX.1-Krea-dev 모델을 사용한 고품질 이미지 생성기**
    
    📝 **Text-to-Image**: 텍스트 프롬프트로 고품질 이미지 생성
    � **Negative Prompt**: 원하지 않는 요소를 제외하여 더 나은 결과 생성
    
    💡 **팁**: Positive prompt에는 원하는 것을, Negative prompt에는 원하지 않는 것을 구체적으로 작성하세요!
    """
    )

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your image description...",
                value="8k, high quality, realistic, high detail, cinematic lighting, a woman walking on a beaching, wearing a red bikini, sunset background, looking at viewer, full body",
                lines=3,
                info="텍스트 프롬프트: 생성하고자 하는 이미지에 대한 상세한 설명을 입력하세요. 구체적이고 명확한 설명일수록 더 나은 결과를 얻을 수 있습니다."
            )
            
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (Optional)",
                placeholder="Enter what you don't want in the image...",
                value="blurry, low quality, distorted, deformed, bad anatomy, bad hands, extra fingers, missing fingers, watermark, text, signature",
                lines=2,
                info="네거티브 프롬프트: 이미지에 포함되지 않았으면 하는 요소들을 입력하세요. 품질 향상에 도움이 됩니다."
            )

            with gr.Row():
                height_input = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Height",
                    info="높이: 생성할 이미지의 세로 크기입니다. 큰 값일수록 더 세밀한 이미지가 생성되지만 처리 시간이 길어집니다. (권장: 1024)",
                )
                width_input = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=64,
                    label="Width",
                    info="너비: 생성할 이미지의 가로 크기입니다. Height와 함께 이미지의 해상도를 결정합니다. (권장: 1024)",
                )

            with gr.Row():
                guidance_scale_input = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=6.5,
                    step=0.1,
                    label="Guidance Scale",
                    info="가이던스 스케일: 프롬프트에 대한 모델의 충실도를 조절합니다. 높은 값은 프롬프트를 더 정확히 따르지만 창의성이 떨어질 수 있습니다. (권장: 3.5-7.5)",
                )
                num_inference_steps_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=30,
                    step=1,
                    label="Inference Steps",
                    info="추론 단계: 이미지 생성을 위한 디노이징 단계 수입니다. 높은 값은 더 정교한 이미지를 생성하지만 처리 시간이 길어집니다. (권장: 20-50)",
                )

            seed_input = gr.Number(
                label="Seed (-1 for random)",
                value=-1,
                precision=0,
                info="시드 값: 이미지 생성의 랜덤성을 제어합니다. 같은 시드 값을 사용하면 동일한 이미지를 재생성할 수 있습니다. -1을 입력하면 랜덤 시드를 사용합니다.",
            )

            generate_btn = gr.Button("Generate Image", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil", height=500)
            output_text = gr.Textbox(
                label="Status",
                interactive=False,
                info="생성 상태 및 결과 정보가 표시됩니다. 사용된 시드 값도 함께 확인할 수 있습니다.",
            )

    # Set up the generate button click event
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            negative_prompt_input,
            height_input,
            width_input,
            guidance_scale_input,
            num_inference_steps_input,
            seed_input,
        ],
        outputs=[output_image, output_text],
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(inbrowser=True)
