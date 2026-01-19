import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
from PIL import Image
import gradio as gr
import random


class ImageEditor:
    def __init__(self):
        self.device = "mps"
        self.dtype = torch.bfloat16
        self.pipe = None

    def load_model(self):
        """Load the model (called once on startup)"""
        if self.pipe is None:
            print("모델 로딩 중...")
            self.pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-4B", torch_dtype=self.dtype
            )
            self.pipe = self.pipe.to(self.device)

            # Memory optimization for macOS
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_attention_slicing(1)
            self.pipe.enable_sequential_cpu_offload()
            print("모델 로딩 완료!")

    def edit_image(
        self,
        input_image,
        prompt,
        guidance_scale=3.5,
        num_inference_steps=20,
        seed=None,
        use_random_seed=True,
    ):
        """
        Edit an image using Klein 4B model (for Gradio)

        Args:
            input_image: PIL Image object from Gradio
            prompt: Text description of desired changes
            guidance_scale: How closely to follow the prompt (1.0-7.0)
            num_inference_steps: Quality vs speed (4-50)
            seed: Random seed for reproducibility
            use_random_seed: If True, ignore seed parameter
        """
        if input_image is None:
            return None, "⚠️ Please upload an image first!"

        if not prompt or prompt.strip() == "":
            return None, "⚠️ Please enter a prompt!"

        # Load model if not already loaded
        self.load_model()

        # Use input image dimensions
        width = input_image.width
        height = input_image.height

        # Handle seed
        if use_random_seed:
            seed = random.randint(0, 2147483647)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        status_msg = f"🎨 Editing image... (Size: {width}x{height}, Steps: {num_inference_steps}, Seed: {seed})"
        print(status_msg)

        try:
            # Generate
            image = self.pipe(
                prompt=prompt,
                image=input_image,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

            # Save
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"edited_{timestamp}.png"
            image.save(output_path)

            success_msg = f"✅ Image edited successfully! Saved as: {output_path}\n"
            success_msg += f"📊 Settings - Guidance: {guidance_scale}, Steps: {num_inference_steps}, Seed: {seed}"
            print(success_msg)

            return image, success_msg

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(error_msg)
            return None, error_msg


def create_ui():
    """Create Gradio interface"""
    editor = ImageEditor()

    with gr.Blocks(title="Flux Klein 4B Image Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 Flux Klein 4B Image Editor
            Upload an image and describe how you want to edit it. The output will maintain the input image dimensions.
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image", type="pil", sources=["upload", "clipboard"]
                )
                prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe the changes you want (e.g., 'turn the sky into sunset colors', 'add snow to the landscape')",
                    lines=3,
                )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=7.0,
                        value=3.5,
                        step=0.5,
                        label="Guidance Scale (higher = closer to prompt)",
                    )
                    num_steps = gr.Slider(
                        minimum=4,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps (higher = better quality, slower)",
                    )

                    with gr.Row():
                        use_random_seed = gr.Checkbox(
                            label="Use Random Seed", value=True
                        )
                        seed = gr.Number(
                            label="Seed (for reproducibility)", value=42, precision=0
                        )

                edit_btn = gr.Button("🚀 Edit Image", variant="primary", size="lg")

            with gr.Column():
                output_image = gr.Image(label="Edited Image")
                status = gr.Textbox(label="Status", lines=3)

        # Examples
        gr.Markdown("### 📝 Example Prompts")
        gr.Examples(
            examples=[
                ["turn the sky into a vibrant sunset with orange and pink clouds"],
                ["make it look like it's snowing"],
                ["change the season to autumn with colorful leaves"],
                ["add dramatic lighting like golden hour"],
                ["convert to black and white film style"],
                ["make it look like a painting in the style of Van Gogh"],
            ],
            inputs=prompt,
            label="Click to use example prompts",
        )

        # Event handler
        edit_btn.click(
            fn=editor.edit_image,
            inputs=[
                input_image,
                prompt,
                guidance_scale,
                num_steps,
                seed,
                use_random_seed,
            ],
            outputs=[output_image, status],
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - The output image will automatically match the input image dimensions
            - Higher guidance scale (5-7) follows the prompt more closely but may look less natural
            - More inference steps (30-50) generally produce better quality but take longer
            - Use a fixed seed for reproducible results
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(inbrowser=True)  # Automatically open in browser
