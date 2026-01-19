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
        negative_prompt="",
        guidance_scale=3.5,
        num_inference_steps=4,
        seed=100,
        use_random_seed=True,
        use_input_dimensions=True,
        custom_width=1024,
        custom_height=1024,
    ):
        """
        Edit an image using Klein 4B model (for Gradio)

        Args:
            input_image: PIL Image object from Gradio
            prompt: Text description of desired changes
            negative_prompt: What to avoid in the image
            guidance_scale: How closely to follow the prompt (1.0-7.0)
            num_inference_steps: Quality vs speed (4-50)
            seed: Random seed for reproducibility
            use_random_seed: If True, ignore seed parameter
            use_input_dimensions: Use input image dimensions
            custom_width: Custom output width
            custom_height: Custom output height
        """
        if input_image is None:
            return None, "⚠️ Please upload an image first!"

        if not prompt or prompt.strip() == "":
            return None, "⚠️ Please enter a prompt!"

        # Load model if not already loaded
        self.load_model()

        # Determine output dimensions
        if use_input_dimensions:
            width = input_image.width
            height = input_image.height
        else:
            width = custom_width
            height = custom_height

        # Handle seed
        if use_random_seed:
            seed = random.randint(0, 2147483647)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        status_msg = f"🎨 Editing image... (Size: {width}x{height}, Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed})"
        print(status_msg)

        try:
            # Prepare pipe parameters
            pipe_params = {
                "prompt": prompt,
                "image": input_image,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }

            # Add negative prompt if provided
            if negative_prompt and negative_prompt.strip():
                pipe_params["negative_prompt"] = negative_prompt

            # Note: Flux2KleinPipeline doesn't support 'strength' parameter
            # The strength slider is kept in UI for potential future use

            # Generate
            image = self.pipe(**pipe_params).images[0]

            # Save
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"edited_{timestamp}.png"
            image.save(output_path)

            success_msg = f"✅ Image edited successfully! Saved as: {output_path}\n"
            success_msg += f"📊 Settings - Size: {width}x{height}, Guidance: {guidance_scale}, Steps: {num_inference_steps}, Seed: {seed}"
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
                    label="Input Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    value="default.jpg",
                    height=700,
                )
                prompt = gr.Textbox(
                    label="Edit Prompt",
                    placeholder="Describe the changes you want (e.g., 'turn the sky into sunset colors', 'add snow to the landscape')",
                    value="8k resolution quality, high detail, high quality, best quality, realistic, masterpiece, cinematic lighting, wearing a dark blue bikini.",
                    lines=3,
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What to avoid (e.g., 'blurry, low quality, distorted')",
                    value="",
                    lines=2,
                )

                with gr.Accordion("⚙️ Generation Settings", open=True):
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.5,
                        step=0.1,
                        label="Guidance Scale",
                        info="How closely to follow the prompt (1.0=creative, 7.0=strict)",
                    )

                    num_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=4,
                        step=1,
                        label="Inference Steps",
                        info="More steps = better quality but slower (4-8 recommended for Klein)",
                    )

                with gr.Accordion("🎲 Seed Settings", open=False):
                    with gr.Row():
                        use_random_seed = gr.Checkbox(
                            label="Use Random Seed", value=False
                        )
                        seed = gr.Number(
                            label="Seed (for reproducibility)", value=100, precision=0
                        )

                with gr.Accordion("📐 Dimension Settings", open=False):
                    use_input_dimensions = gr.Checkbox(
                        label="Use Input Image Dimensions",
                        value=True,
                        info="When checked, output will match input size",
                    )

                    with gr.Row():
                        custom_width = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="Custom Width",
                            info="Only used when 'Use Input Dimensions' is unchecked",
                        )
                        custom_height = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="Custom Height",
                            info="Only used when 'Use Input Dimensions' is unchecked",
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
                negative_prompt,
                guidance_scale,
                num_steps,
                seed,
                use_random_seed,
                use_input_dimensions,
                custom_width,
                custom_height,
            ],
            outputs=[output_image, status],
        )

        gr.Markdown(
            """
            ---
            ### 💡 Tips & Guidelines

            **Prompts:**
            - Be descriptive and specific about what you want
            - Use negative prompts to avoid unwanted elements

            **Generation Settings:**
            - **Guidance Scale**: 3.5 is balanced, 1.0 is creative/loose, 7.0+ is strict/literal
            - **Inference Steps**: Klein works well with just 4-8 steps (unlike other models)

            **Seed Settings:**
            - Uncheck "Use Random Seed" and set a specific number for reproducible results
            - Same seed + same settings = same output every time

            **Dimensions:**
            - By default, output matches input dimensions (recommended)
            - Custom dimensions may affect aspect ratio and composition
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(inbrowser=True)  # Automatically open in browser
