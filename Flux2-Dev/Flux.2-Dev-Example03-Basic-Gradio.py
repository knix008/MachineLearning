import torch
from diffusers import Flux2Pipeline
from datetime import datetime
import os
import gradio as gr

repo_id = "black-forest-labs/FLUX.2-dev"
if torch.cuda.is_available():
    device = "cuda"
    torch_dtype = torch.float16
else:
    device = "cpu"
    torch_dtype = torch.float32

# Load model
pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
print("Model loaded!")

default_prompt = "4k, high-quality, high resolution, realistic, photography of a beautiful Instagram-style korean girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner.Her eyes are hazel with a natural sparkle of happiness as she smiles, wearing a red bikini.The image should be perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

#default_prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner.Her eyes are hazel with a natural sparkle of happiness as she smiles.The image should be perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

#default_prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

device_for_generator = "cuda" if torch.cuda.is_available() else "cpu"
script_name = os.path.splitext(os.path.basename(__file__))[0]


def generate_image(prompt, height, width, num_steps, guidance_scale, seed):
    generator = torch.Generator(device=device_for_generator).manual_seed(int(seed))

    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        generator=generator,
        num_inference_steps=int(num_steps),
        guidance_scale=guidance_scale,
    ).images[0]

    # Save with program name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{script_name}_{timestamp}.png"
    image.save(filename)

    return image, f"Saved: {filename}"


with gr.Blocks(title="Flux.2-Dev Image Generator") as demo:
    gr.Markdown("# Flux.2-Dev Image Generator")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                info="Describe the image you want to generate",
                placeholder="Enter your prompt here...",
                value=default_prompt,
                lines=3,
            )
            with gr.Row():
                height = gr.Slider(
                    256,
                    2048,
                    value=1024,
                    step=64,
                    label="Height",
                    info="Image height in pixels",
                )
                width = gr.Slider(
                    256,
                    2048,
                    value=512,
                    step=64,
                    label="Width",
                    info="Image width in pixels",
                )
            with gr.Row():
                num_steps = gr.Slider(
                    1,
                    50,
                    value=28,
                    step=1,
                    label="Steps",
                    info="More steps = higher quality but slower",
                )
                guidance_scale = gr.Slider(
                    0,
                    20,
                    value=4,
                    step=0.5,
                    label="Guidance Scale",
                    info="How closely to follow the prompt (higher = more strict)",
                )
            seed = gr.Number(
                value=42, label="Seed", info="Random seed for reproducible results"
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            status = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_steps, guidance_scale, seed],
        outputs=[output_image, status],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
