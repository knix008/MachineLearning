from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import gradio as gr

model_id = "stabilityai/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16,
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, transformer=model_nf4, torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

def generate_image(
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    max_sequence_length,
    seed
):
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None and seed != -1:
        generator = generator.manual_seed(seed)
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        generator=generator,
    )
    return result.images[0]

demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(lines=4, label="Prompt", value="A whimsical and creative image depicting a hybrid creature..."),
        gr.Textbox(lines=2, label="Negative Prompt", value="blurry, low quality, distorted"),
        gr.Slider(1, 100, value=28, step=1, label="Num Inference Steps"),
        gr.Slider(1.0, 20.0, value=4.5, step=0.1, label="Guidance Scale"),
        gr.Slider(64, 4096, value=512, step=64, label="Max Sequence Length"),
        gr.Number(label="Seed (set -1 for random)", value=-1, precision=0),
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Stable Diffusion 3.5 Large - Custom Parameters",
    description="입력값을 조정해 이미지를 생성하세요."
)

if __name__ == "__main__":
    demo.launch()
