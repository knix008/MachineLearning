import torch
from diffusers import AutoPipelineForImage2Image
import gradio as gr
import datetime

# Load prior pipeline for embedding generation

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
print("> Model loaded...")


# 이미지 생성 함수 구현
def generate_image(
    prompt,
    negative_prompt,
    input_image,
    num_inference_steps,
    guidance_scale,
    seed,
):
    generator = torch.Generator("cpu").manual_seed(int(seed))

    # decoder pipeline에서 이미지 생성
    image = pipeline(
        prompt=prompt,
        image=input_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        callback_on_step_end=None,
        generator=generator,
    ).images[0]

    image.save(
        f"Kandinsky-Example03-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    )
    return image


with gr.Blocks() as demo:
    gr.Markdown("# Kandinsky Image2Image Demo with Prior Embeddings")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 입력 이미지 (Upload or Clipboard)")
            input_image = gr.Image(
                label="입력 이미지 (Upload or Clipboard)",
                type="pil",
                sources=["upload", "clipboard"],
                value="default,jpg"
                height=500,
            )
            gr.Markdown("## 프롬프트 설정")
            prompt = gr.Textbox(
                label="Prompt",
                value="8k, high detail, high quality, photo realistic, masterpiece, best quality, dark blue bikini",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="low resolution, bad anatomy, bad hands, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature",
            )
            gr.Markdown("## 생성 설정")
            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=1, maximum=100, value=30, step=1
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.1
            )
            seed = gr.Number(label="Seed", value=42)
            generate_btn = gr.Button("Generate")
        with gr.Column():
            gr.Markdown("## 결과 이미지")
            output_image = gr.Image(label="Generated Image", height=500)

    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt,
            negative_prompt,
            input_image,
            num_inference_steps,
            guidance_scale,
            seed,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
