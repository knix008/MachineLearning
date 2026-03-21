import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
import datetime
import cv2
from PIL import Image

#  Please import it from `diffusers.models.transformers.transformer_flux`?

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

def make_control_image(input_path, output_path):
    """
    Create a control image from the input image using Canny edge detection.
    """
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    Image.fromarray(edges).save(output_path)
    return load_image(output_path)

input_path = "default.jpg"  # Path to your input image
output_path = "default_canny.png"  # Path to save the control image

control_image = make_control_image(input_path, output_path)
width, height = control_image.size

prompt = "A young girl stands gracefully at the edge of a serene beach, her long, flowing hair gently tousled by the sea breeze. She wears a soft, pastel-colored dress that complements the tranquil blues and greens of the coastal scenery. The golden hues of the setting sun cast a warm glow on her face, highlighting her serene expression. The background features a vast, azure ocean with gentle waves lapping at the shore, surrounded by distant cliffs and a clear, cloudless sky. The composition emphasizes the girl's serene presence amidst the natural beauty, with a balanced blend of warm and cool tones."

image = pipe(
    prompt,
    control_image=control_image,
    width=width,
    height=height,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.8,
    num_inference_steps=30,
    guidance_scale=3.5,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save(f"flux_controlnet_union_pro_example-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
print("Image generation complete. Check the saved image.")