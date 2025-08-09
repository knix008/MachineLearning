import torch
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid
import datetime
import gradio as gr

# Load prior pipeline for embedding generation

prior_pipeline = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
prior_pipeline.enable_model_cpu_offload()

# 이미지 생성 함수 구현
def generate_image(
    prompt,
    negative_prompt,
    input_image,
    num_inference_steps,
    guidance_scale,
    prior_strength,
    negative_prior_prompt,
    negative_prior_strength,
    seed,
):
    import torch
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    # prior pipeline에서 임베딩 생성
    img_emb = prior_pipeline(
        prompt=prompt,
        image=input_image,
        strength=prior_strength,
        generator=generator
    ).images[0]
    negative_emb = prior_pipeline(
        prompt=negative_prior_prompt,
        image=input_image,
        strength=negative_prior_strength,
        generator=generator
    ).images[0]
    # decoder pipeline에서 이미지 생성
    image = pipeline(
        prompt,
        image=input_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        callback_on_step_end=None,
        # img_emb=img_emb,
        # negative_emb=negative_emb
    ).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# Kandinsky Image2Image Demo with Prior Embeddings")
    with gr.Group():
        gr.Markdown("## 1. 입력 이미지")
        input_image = gr.Image(
            label="입력 이미지 (Upload or Clipboard)",
            type="pil",
            sources=["upload", "clipboard"],
            height=500,
            value="default.jpg"
        )
    with gr.Group():
        gr.Markdown("## 2. 프롬프트 설정")
        prompt = gr.Textbox(
            label="Prompt",
            value="8k, high detail, high quality, realistic, masterpiece, best quality, dark blue bikini",
        )
        negative_prompt = gr.Textbox(
            label="Negative Prompt",
            value="low resolution, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        )
    with gr.Group():
        gr.Markdown("## 3. 생성 설정")
        with gr.Row():
            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=100, value=30, step=1
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=20.0, value=4.0, step=0.1
            )
            seed = gr.Number(label="Seed", value=43)
    with gr.Group():
        gr.Markdown("## 4. Prior 설정")
        with gr.Row():
            prior_strength = gr.Slider(
                label="Prior Strength (positive)",
                minimum=0.0,
                maximum=1.0,
                value=0.85,
                step=0.01,
            )
            negative_prior_strength = gr.Slider(
                label="Prior Strength (negative)",
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.01,
            )
        negative_prior_prompt = gr.Textbox(
            label="Negative Prior Prompt",
            value="low resolution, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
        )
    with gr.Group():
        gr.Markdown("## 5. 결과 이미지")
        output_image = gr.Image(label="Generated Image", height=350)
        generate_btn = gr.Button("Generate")


    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            input_image,
            num_inference_steps,
            guidance_scale,
            prior_strength,
            negative_prior_prompt,
            negative_prior_strength,
            seed,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
