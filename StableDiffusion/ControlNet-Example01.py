import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import requests
from io import BytesIO
import base64


class ControlNetImageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device="cuda"):
        """
        Initialize ControlNet pipeline

        Args:
            model_id: Stable Diffusion model to use
            device: Device to run on (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load ControlNet model for Canny edge detection (SD 2.1 compatible)
        self.controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-canny-diffusers",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        # Load Stable Diffusion pipeline with ControlNet
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        self.pipe = self.pipe.to(self.device)

        # Enable memory efficient attention if available
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except:
                print("xformers not available, using default attention")

    def create_canny_edge(self, image, low_threshold=100, high_threshold=200):
        """
        Create Canny edge detection from input image

        Args:
            image: PIL Image or numpy array
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection

        Returns:
            PIL Image of Canny edges
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply Canny edge detection
        canny = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert back to PIL Image
        canny_image = Image.fromarray(canny)

        return canny_image

    def create_simple_sketch(self, width=768, height=768):
        """
        Create a simple sketch for demonstration (SD 2.1 default resolution)

        Returns:
            PIL Image of a simple sketch
        """
        # Create a white canvas
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Draw a simple house (scaled for 768x768)
        # House base
        draw.rectangle([225, 450, 525, 675], outline="black", width=4)

        # Roof
        draw.polygon([(195, 450), (375, 300), (555, 450)], outline="black", width=4)

        # Door
        draw.rectangle([330, 570, 420, 675], outline="black", width=3)

        # Windows
        draw.rectangle([255, 480, 315, 540], outline="black", width=3)
        draw.rectangle([435, 480, 495, 540], outline="black", width=3)

        # Convert to grayscale and then to canny-like edges
        gray = img.convert("L")
        return gray.convert("RGB")

    def resize_image_with_aspect_ratio(self, image, max_size=768):
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image to resize
            max_size: Maximum dimension size
            
        Returns:
            Resized PIL Image with maintained aspect ratio
        """
        # Get original dimensions
        width, height = image.size
        
        # Calculate scaling factor
        if width > height:
            # Landscape or square
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            # Portrait
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        # Ensure dimensions are multiples of 8 (requirement for Stable Diffusion)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Minimum size constraints
        new_width = max(new_width, 512)
        new_height = max(new_height, 512)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def generate_image(
        self,
        prompt,
        control_image,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        seed=None,
        width=None,
        height=None,
    ):
        """
        Generate image using ControlNet

        Args:
            prompt: Text prompt for generation
            control_image: PIL Image for control guidance
            negative_prompt: Negative prompt to avoid certain features
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            controlnet_conditioning_scale: How strongly to follow control image
            seed: Random seed for reproducibility
            width: Output image width (optional)
            height: Output image height (optional)

        Returns:
            Generated PIL Image
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Use control image dimensions if width/height not specified
        if width is None or height is None:
            width, height = control_image.size

        # Generate image
        image = self.pipe(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            height=height,
            width=width,
        ).images[0]

        return image

    def visualize_results(self, original_image, control_image, generated_image, prompt):
        """
        Return images for Gradio display
        """
        return original_image, control_image, generated_image


# Global variable to store the generator
generator = None


def initialize_model_on_startup():
    """Initialize the ControlNet model on startup"""
    global generator
    print("🔄 Initializing ControlNet model...")
    try:
        generator = ControlNetImageGenerator()
        print("✅ Model initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        return False


def generate_with_controlnet(
    input_image,
    prompt,
    negative_prompt,
    num_steps,
    guidance_scale,
    controlnet_scale,
    seed,
    canny_low,
    canny_high,
):
    """Generate image using ControlNet with Gradio interface"""
    global generator

    if generator is None:
        return (
            None,
            None,
            "❌ Model initialization failed. Please restart the application.",
        )

    try:
        # Handle the case where no image is uploaded
        if input_image is None:
            # Create a simple sketch
            input_image = generator.create_simple_sketch()
            control_image = input_image
            target_width, target_height = 768, 768
            status = "ℹ️ No image uploaded, using default sketch"
        else:
            # Resize input image while maintaining aspect ratio
            resized_image = generator.resize_image_with_aspect_ratio(input_image)
            target_width, target_height = resized_image.size
            
            # Create Canny edge detection
            control_image = generator.create_canny_edge(
                resized_image, int(canny_low), int(canny_high)
            )
            status = f"✅ Canny edges created from uploaded image (resized to {target_width}x{target_height})"

        # Generate the image with original aspect ratio
        generated_image = generator.generate_image(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_scale),
            seed=int(seed) if seed != -1 else None,
            width=target_width,
            height=target_height,
        )

        return control_image, generated_image, status

    except Exception as e:
        return None, None, f"❌ Error generating image: {str(e)}"


def create_sample_sketch():
    """Create and return a sample sketch"""
    global generator
    if generator is None:
        # Create a temporary generator just for the sketch
        temp_gen = ControlNetImageGenerator.__new__(ControlNetImageGenerator)
        sketch = temp_gen.create_simple_sketch()
    else:
        sketch = generator.create_simple_sketch()
    return sketch


def create_gradio_interface():
    """Create the Gradio interface"""

    # Get model status
    model_status = (
        "✅ Model initialized and ready!"
        if generator is not None
        else "❌ Model failed to initialize"
    )

    with gr.Blocks(title="ControlNet Image Generator", theme=gr.themes.Soft()) as iface:
        gr.Markdown(
            """
        # 🎨 ControlNet Image Generator
        
        Generate AI images with precise control using ControlNet and Stable Diffusion.
        Upload an image or use the default sketch to guide the generation process.
        
        **Instructions:**
        1. Upload an image or use the default sketch
        2. Enter your prompt and adjust parameters
        3. Click "Generate Image"
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Model status display
                gr.Textbox(label="Model Status", value=model_status, interactive=False)

                gr.Markdown("### Input Controls")

                # Image input
                input_image = gr.Image(
                    label="Upload Image (optional)", type="pil", height=300
                )

                sample_btn = gr.Button("📝 Use Sample Sketch", variant="secondary")

                # Text inputs
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="a beautiful landscape with mountains and lake, high quality, detailed",
                    lines=3,
                    value="dark blue bikini, skinny, perfect anatomy, good skin, good hair, good fingers, good legs, photorealistic, ultra high definition, ultra high resolution,8k resolution, ultra detail, vibrant colors, cinematic lighting, realistic shadows, high quality, masterpiece, best quality",
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted",
                    lines=2,
                    value="blurring, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username",
                )

                # Parameters
                with gr.Accordion("⚙️ Advanced Parameters", open=False):
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Number of Steps",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale",
                    )
                    controlnet_scale = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="ControlNet Conditioning Scale",
                    )
                    seed = gr.Number(
                        label="Seed (-1 for random)", value=-1, precision=0
                    )

                    with gr.Row():
                        canny_low = gr.Slider(
                            minimum=50,
                            maximum=150,
                            value=100,
                            step=10,
                            label="Canny Low Threshold",
                        )
                        canny_high = gr.Slider(
                            minimum=150,
                            maximum=300,
                            value=200,
                            step=10,
                            label="Canny High Threshold",
                        )

                # Generate button
                generate_btn = gr.Button(
                    "🎨 Generate Image", variant="primary", size="lg"
                )

            with gr.Column(scale=2):
                gr.Markdown("### Results")

                status_output = gr.Textbox(label="Generation Status", interactive=False)

                with gr.Row():
                    control_output = gr.Image(
                        label="Control Image (Canny Edges)", height=300
                    )
                    generated_output = gr.Image(label="Generated Image", height=300)

        # Event handlers
        sample_btn.click(fn=create_sample_sketch, outputs=[input_image])

        generate_btn.click(
            fn=generate_with_controlnet,
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                controlnet_scale,
                seed,
                canny_low,
                canny_high,
            ],
            outputs=[control_output, generated_output, status_output],
        )

        # Examples
        gr.Examples(
            examples=[
                [
                    None,
                    "a futuristic cityscape with neon lights, cyberpunk style, high quality",
                    "blurry, low quality, distorted",
                    20,
                    7.5,
                    1.0,
                    42,
                    100,
                    200,
                ],
                [
                    None,
                    "a magical forest with glowing trees, fantasy art style",
                    "blurry, low quality",
                    25,
                    8.0,
                    1.2,
                    123,
                    80,
                    180,
                ],
                [
                    None,
                    "a steampunk mansion with brass details and gears, vintage photography",
                    "modern, clean, minimalist",
                    30,
                    9.0,
                    0.8,
                    456,
                    120,
                    220,
                ],
            ],
            inputs=[
                input_image,
                prompt,
                negative_prompt,
                num_steps,
                guidance_scale,
                controlnet_scale,
                seed,
                canny_low,
                canny_high,
            ],
        )

        gr.Markdown(
            """
        ### 💡 Tips for Better Results:
        - **Clear prompts**: Be specific about what you want to see
        - **Control strength**: Lower values (0.5-0.8) give more creative freedom, higher values (1.0-1.5) follow edges more strictly
        - **Steps**: More steps generally mean better quality but take longer
        - **Guidance scale**: Higher values follow the prompt more closely
        - **Canny thresholds**: Adjust these to control edge detection sensitivity
        
        ### 🎯 Available ControlNet Models:
        - **Canny**: Edge detection (current)
        - **Depth**: Depth maps
        - **OpenPose**: Human poses  
        - **Scribble**: Hand-drawn scribbles
        - **Semantic Segmentation**: Object boundaries
        """
        )

    return iface


def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Starting ControlNet Gradio Interface...")
    print("=" * 50)

    # Initialize model on startup
    model_loaded = initialize_model_on_startup()

    if not model_loaded:
        print("⚠️  Model initialization failed, but interface will still launch.")
        print("Some features may not work properly.")

    # Create and launch interface
    interface = create_gradio_interface()

    # Launch the interface
    interface.launch(
        share=False,  # Set to True if you want to create a public link
        inbrowser=True,  # Automatically open in browser
        server_name="127.0.0.1",  # Local access only
        server_port=7860,  # Port number
        show_error=True,
    )


def demo_other_controlnet_models():
    """
    Information about other ControlNet models available
    """
    models_info = {
        "Canny": "lllyasviel/sd-controlnet-canny - Edge detection (currently used)",
        "Depth": "lllyasviel/sd-controlnet-depth - Depth maps for 3D structure",
        "OpenPose": "lllyasviel/sd-controlnet-openpose - Human pose detection",
        "Scribble": "lllyasviel/sd-controlnet-scribble - Hand-drawn scribbles",
        "HED": "lllyasviel/sd-controlnet-hed - Holistically-nested edge detection",
        "Semantic Seg": "lllyasviel/sd-controlnet-seg - Semantic segmentation",
        "Normal": "lllyasviel/sd-controlnet-normal - Surface normal maps",
        "MLSeg": "lllyasviel/sd-controlnet-mlsd - Line segment detection",
    }

    print("\n" + "=" * 60)
    print("Available ControlNet Models:")
    print("=" * 60)
    for name, description in models_info.items():
        print(f"• {name}: {description}")

    print("\nTo use a different model, modify the ControlNetImageGenerator class:")
    print("Replace 'lllyasviel/sd-controlnet-canny' with the desired model ID")


if __name__ == "__main__":
    main()
