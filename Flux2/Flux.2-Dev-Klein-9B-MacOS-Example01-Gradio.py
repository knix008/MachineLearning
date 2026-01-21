import torch
import gradio as gr
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os

device = "mps"
dtype = torch.bfloat16

# Initialize model once
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype
)
pipe = pipe.to(device)

pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.enable_attention_slicing(1)  # reduce memory usage further
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")


def generate_image(prompt, height, width, guidance_scale, num_inference_steps, seed):
    """Generate image based on parameters"""
    try:
        image = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
            generator=torch.Generator(device=device).manual_seed(int(seed)),
        ).images[0]

        # Get script filename without extension
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{script_name}_{timestamp}.png"
        image.save(filename)
        print(f"이미지가 저장되었습니다: {filename}")

        return image, f"✓ 이미지 생성 완료: {filename}"
    except Exception as e:
        return None, f"✗ 오류 발생: {str(e)}"


# Create Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="프롬프트 (Prompt)",
            value="A dreamy aesthetic portrait of a girl with pastel tones, soft glowing skin, gentle lighting, whimsical background of blooming flowers, cinematic bokeh, modern Pinterest-inspired look, natural face preserved, elegant and artistic Instagram photo style, 9:16 aspect ratio.",
            lines=5,
        ),
        gr.Slider(
            label="높이 (Height)", minimum=256, maximum=1024, step=64, value=1024
        ),
        gr.Slider(label="너비 (Width)", minimum=256, maximum=1024, step=64, value=768),
        gr.Slider(
            label="가이던스 스케일 (Guidance Scale)",
            minimum=0.0,
            maximum=10.0,
            step=0.1,
            value=1.0,
        ),
        gr.Slider(
            label="추론 스텝 (Inference Steps)", minimum=1, maximum=50, step=1, value=28
        ),
        gr.Number(label="시드 (Seed)", value=400),
    ],
    outputs=[
        gr.Image(label="생성된 이미지 (Generated Image)", type="pil", height=600),
        gr.Textbox(label="상태 (Status)"),
    ],
    title="FLUX.2 Klein 9B 이미지 생성기 (Image Generator)",
    description="매개변수를 설정하여 AI 이미지를 생성합니다. (Set parameters to generate AI images)",
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
