import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import time

# 모델 로딩 (Stable Diffusion 2.1로 변경)
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(
    reference_image, prompt, strength=0.8, guidance_scale=7.5, num_inference_steps=50
):
    start_time = time.time()
    if reference_image is None:
        return None, "참조 이미지를 업로드해 주세요."
    if prompt.strip() == "":
        return None, "프롬프트를 입력해 주세요."
    image = reference_image.convert("RGB").resize((512, 512))

    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        result = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
    end_time = time.time()
    elapsed = end_time - start_time
    return result.images[0], f"생성 소요 시간: {elapsed:.2f}초"


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Reference Image 생성기")
    with gr.Row():
        with gr.Column():
            ref_img = gr.Image(label="참조 이미지", type="pil")
            prompt = gr.Textbox(
                label="프롬프트(설명)", placeholder="예: a cat wearing sunglasses"
            )
            strength = gr.Slider(
                0.1, 1.0, value=0.8, step=0.05, label="참조 반영 강도(strength)"
            )
            guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance scale")
            steps = gr.Slider(10, 100, value=50, step=1, label="Inference steps")
            btn = gr.Button("이미지 생성")
        with gr.Column():
            output_img = gr.Image(label="생성된 이미지")
            elapsed_label = gr.Textbox(label="걸린 시간/안내", interactive=False)

    btn.click(
        generate_image,
        inputs=[ref_img, prompt, strength, guidance, steps],
        outputs=[output_img, elapsed_label],
    )

if __name__ == "__main__":
    demo.launch()
