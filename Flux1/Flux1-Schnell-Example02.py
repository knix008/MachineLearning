import torch
from diffusers import DiffusionPipeline
from PIL import Image
import gradio as gr
import gc
import datetime

# 모델 로딩 및 최적화
torch.cuda.empty_cache()

model_id = "black-forest-labs/FLUX.1-schnell"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
print("> 모델 로딩 완료!")

try:
    pipe.enable_xformers_memory_efficient_attention()
except (ImportError, AttributeError):
    try:
        pipe.unet.set_attn_processor(None)
    except Exception:
        pass


def generate_image(prompt, negative_prompt, num_inference_steps, guidance_scale):
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
    # 결과 이미지 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result.save(f"output_{timestamp}.png")
    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()
    return result


description = """
# FLUX.1 Schnell 이미지 생성기 (Gradio)
Stable Diffusion 기반 FLUX.1 Schnell 모델을 사용하여 이미지를 생성합니다.  
아래 입력값을 조정해 원하는 이미지를 만들어보세요.

- **Prompt**: 생성하고 싶은 이미지에 대한 설명을 입력하세요.
- **Negative Prompt**: 생성에서 제외하고 싶은 요소를 입력하세요.
- **Inference Steps**: 이미지 생성 품질과 속도에 영향을 줍니다. (값이 높을수록 품질↑, 속도↓)
- **Guidance Scale**: 프롬프트 반영 강도입니다. (값이 높을수록 프롬프트에 더 충실)
"""

with gr.Blocks() as demo:
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="8k, a skinny woman photo, walking on a sunset beach, wearing a dark blue bikini with high leg style, high detail, full body figure, cinematic lighting, photo realistic",
                info="생성하고 싶은 이미지를 설명하세요.",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, cartoon, ugly, bad anatomy, bad hands, text, watermark",
                info="제외하고 싶은 요소를 입력하세요.",
            )
            num_inference_steps = gr.Slider(
                minimum=10,
                maximum=50,
                value=25,
                step=1,
                label="Inference Steps",
                info="이미지 생성 반복 횟수 (품질/속도 조절)",
            )
            guidance_scale = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                value=7.5,
                step=0.1,
                label="Guidance Scale",
                info="프롬프트 반영 강도",
            )
            btn = gr.Button("이미지 생성")
        with gr.Column():
            output_image = gr.Image(label="생성된 이미지", type="pil", height=512)

    btn.click(
        fn=generate_image,
        inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
