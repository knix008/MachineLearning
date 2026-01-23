import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os
import gradio as gr

device = "mps"
dtype = torch.float16  # Use float16 instead of bfloat16 for MPS compatibility

#prompt_input = "A highly realistic, 4k, high-quality vivid photo of a beautiful skinny girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, wearing a red bikini, looking at the sea. The image should capture her in a full-body shot with perfect anatomy including precise details in her eyes and teeth. Her skin should appear natural, avoiding an overly smooth or filtered look, to maintain a lifelike. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

# Initialize the pipeline
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype)
pipe = pipe.to(device)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

def generate_image(prompt, height, width, num_inference_steps, seed):
    """Generate image based on user input parameters"""
    try:
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(seed)
        ).images[0]
        
        # Save with filename and datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = os.path.splitext(os.path.basename(__file__))[0]
        output_path = f"{base_name}_{timestamp}.png"
        image.save(output_path)
        
        return image, f"Image saved as: {output_path}"
    except Exception as e:
        return None, f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Flux 2 Klein 4B Image Generator") as demo:
    gr.Markdown("# Flux 2 Klein 4B Image Generator")
    
    with gr.Row():
        with gr.Column():
            # Input components
            prompt = gr.Textbox(
                label="Prompt",
                value="A highly realistic, 4k, high-quality vivid photo of a beautiful skinny animation-style girl. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, wearing a red bikini. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, avoiding an overly smooth or filtered look, to maintain a lifelike. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood.",
                lines=4,
                placeholder="Enter your prompt here"
            )
            
            with gr.Row():
                height = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    value=1024,
                    step=256,
                    label="Height"
                )
                width = gr.Slider(
                    minimum=256,
                    maximum=1536,
                    value=1024,
                    step=256,
                    label="Width"
                )
            
            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                    label="Inference Steps"
                )
                seed = gr.Slider(
                    minimum=0,
                    maximum=2147483647,
                    value=0,
                    step=1,
                    label="Seed"
                )
            
            generate_btn = gr.Button("Generate Image", variant="primary")
        
        with gr.Column():
            # Output components
            output_image = gr.Image(label="Generated Image", type="pil")
            output_message = gr.Textbox(label="Status", interactive=False)
    
    # Connect button to generation function
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, num_inference_steps, seed],
        outputs=[output_image, output_message]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(inbrowser=True)