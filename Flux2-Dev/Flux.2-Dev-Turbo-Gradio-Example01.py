import torch
import gradio as gr
from diffusers import Flux2Pipeline
from datetime import datetime

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

device = "cuda"
device_type = torch.bfloat16

print("모델 로딩 중...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=device_type
).to(device)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)
print("모델 로딩 완료!")


def generate_image(prompt, guidance_scale, height, width, num_steps, seed):
    if seed == -1:
        actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    else:
        actual_seed = int(seed)

    generator = torch.Generator(device="cpu").manual_seed(actual_seed)

    image = pipe(
        prompt=prompt,
        sigmas=TURBO_SIGMAS[:num_steps] if num_steps <= len(TURBO_SIGMAS) else None,
        guidance_scale=guidance_scale,
        height=int(height),
        width=int(width),
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]

    # Save image with timestamp, steps, seed, and guidance scale
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"flux2_turbo_{timestamp}_steps{int(num_steps)}_seed{actual_seed}_guidance{guidance_scale}.jpg"
    image.save(output_filename)

    return image, f"저장됨: {output_filename}"


with gr.Blocks(title="FLUX.2-dev Turbo") as demo:
    gr.Markdown("# FLUX.2-dev Turbo Image Generator")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=4,
                value="A highly realistic, high-quality photo of a beautiful Instagram-style korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a half-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."
            )

            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=10.0, value=2.5, step=0.1,
                    label="Guidance Scale"
                )
                num_steps = gr.Slider(
                    minimum=4, maximum=20, value=8, step=1,
                    label="Inference Steps"
                )

            with gr.Row():
                width = gr.Slider(
                    minimum=512, maximum=1536, value=768, step=64,
                    label="Width"
                )
                height = gr.Slider(
                    minimum=512, maximum=1536, value=1024, step=64,
                    label="Height"
                )

            seed = gr.Number(label="Seed (-1 for random)", value=-1)

            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Generated Image", type="pil")
            status = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, guidance_scale, height, width, num_steps, seed],
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    demo.launch(share=False)
