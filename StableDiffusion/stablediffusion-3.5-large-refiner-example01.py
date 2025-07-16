import torch
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline
import gradio as gr
import time


def generate_high_resolution_image(prompt, resolution=(1024, 1024)):
    base = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    )
    base.enable_model_cpu_offload()
    print("> Base Model loaded successfully.")
    base.to("cpu")

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.enable_model_cpu_offload()
    print("> Refiner Model loaded successfully.")
    refiner.to("cpu")

    n_steps = 40
    high_noise_frac = 0.8
    seed = -1
    # 시드 설정
    generator = None
    if seed is not None and int(seed) != -1:
        generator = torch.Generator(device=base.device).manual_seed(int(seed))
    
    # Extract width and height from resolution parameter
    width, height = resolution
        
    image = base(
        prompt=prompt,
        negative_prompt=None,
        num_inference_steps=n_steps,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        height=resolution[1],
        width=resolution[0],
    ).images[0]

    return image


def gradio_interface(prompt, width, height):
    start_time = time.time()
    image = generate_high_resolution_image(prompt, resolution=(width, height))
    elapsed = time.time() - start_time
    elapsed_str = f"이미지 생성 소요 시간: {elapsed:.2f}초"
    return image, elapsed_str


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion 3.5 Large + Refine 고해상도 이미지 생성기")
    with gr.Row():
        prompt = gr.Textbox(
            label="프롬프트",
            value="a beautiful skinny woman wearing a high legged red bikini, walking on the sunny beach, photorealistic, 8k resolution, ultra detailed, vibrant colors, highly detailed, cinematic lighting, realistic shadows, depth of field, no bad anatomy, no text, no watermark, no logo, no signature, no low quality, no blurry, no bad quality, no low resolution, no cropped image",
        )
    with gr.Row():
        width = gr.Slider(512, 2048, value=1024, step=64, label="가로 해상도")
        height = gr.Slider(512, 2048, value=1024, step=64, label="세로 해상도")
    with gr.Row():
        btn = gr.Button("이미지 생성")
    with gr.Row():
        output = gr.Image(label="생성된 이미지", type="pil")
    with gr.Row():
        time_output = gr.Textbox(label="소요 시간", interactive=False)

    btn.click(
        fn=gradio_interface,
        inputs=[prompt, width, height],
        outputs=[output, time_output],
    )

if __name__ == "__main__":
    demo.launch()
