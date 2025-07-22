import torch
from diffusers import DiffusionPipeline
import gradio as gr
import time

# 모델 로딩 (최초 1회만)
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
pipe.enable_sequential_cpu_offload()


def generate_image(prompt, guidance_scale, height, width, steps, max_seq_len):
    start_time = time.time()
    output = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=steps,
        max_sequence_length=max_seq_len,
    )
    elapsed = time.time() - start_time
    img = output.images[0]
    mem_info = ""
    if torch.cuda.is_available():
        mem_info = (
            f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )
    else:
        mem_info = "CUDA is not available."
    result_text = f"{mem_info}\nElapsed time: {elapsed:.2f} seconds"
    return img, result_text


with gr.Blocks() as demo:
    gr.Markdown("## Astronaut Image Generator (Diffusers + Gradio)")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                value="An astronaut riding a horse on Mars", label="Prompt"
            )
            scale = gr.Slider(0.0, 10.0, value=7.5, step=0.1, label="Guidance Scale")
            height = gr.Slider(256, 1024, step=8, value=768, label="Image Height")
            width = gr.Slider(256, 2048, step=8, value=1360, label="Image Width")
            steps = gr.Slider(1, 50, value=4, step=1, label="Inference Steps")
            max_seq = gr.Slider(32, 512, value=256, step=8, label="Max Sequence Length")
            btn = gr.Button("Generate")
        with gr.Column():
            output_img = gr.Image(label="Output Image")
            output_txt = gr.Textbox(label="Status and Elapsed Time")
    btn.click(
        fn=generate_image,
        inputs=[prompt, scale, height, width, steps, max_seq],
        outputs=[output_img, output_txt],
    )

if __name__ == "__main__":
    demo.launch()
