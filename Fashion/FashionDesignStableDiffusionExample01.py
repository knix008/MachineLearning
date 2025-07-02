import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
import time

# Stable Diffusion 2.1 모델 로드
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_fashion(prompt, num_inference_steps=30, guidance_scale=7.5, seed=None):
    start_time = time.time()
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        generator = generator.manual_seed(int(seed))
    # 프롬프트에 배경 흰색 조건을 추가
    prompt_full = (
        f"{prompt}, full body fashion design, plain white background, no shadow, no background object"
    )
    negative_prompt = "background, scenery, shadow, text, watermark"
    with torch.inference_mode():
        image = pipe(
            prompt=prompt_full,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator
        ).images[0]
    elapsed = time.time() - start_time
    return image, f"{elapsed:.2f}초"

with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion 2.1 기반 패션디자인 생성기 (배경: 흰색)")
    gr.Markdown(
        "프롬프트(설명어)를 입력하면 Stable Diffusion 2.1로 **흰색 배경** 위에 패션디자인 이미지를 생성합니다.<br>생성에 걸린 시간도 함께 표시됩니다."
    )
    with gr.Row():
        prompt = gr.Textbox(label="패션 프롬프트(예: 'modern dress, runway')", value="fashion illustration, futuristic dress, high detail")
    with gr.Row():
        num_steps = gr.Slider(10, 100, value=30, label="추론 스텝 수 (권장: 30~50)")
        guidance = gr.Slider(1, 20, value=7.5, step=0.1, label="Guidance Scale")
        seed = gr.Number(value=None, label="시드(고정값, 옵션)", precision=0)
    btn = gr.Button("패션 디자인 생성")
    with gr.Row():
        output_img = gr.Image(label="생성된 패션 디자인(흰색 배경)")
        elapsed_time = gr.Text(label="소요 시간")
    btn.click(
        generate_fashion,
        inputs=[prompt, num_steps, guidance, seed],
        outputs=[output_img, elapsed_time]
    )

if __name__ == "__main__":
    demo.launch()