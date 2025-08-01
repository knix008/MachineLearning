import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
import datetime

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora", adapter_name="canny")
pipe.set_adapters("canny", 0.85)

# Model loading and configuration
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()
pipe.enable_model_cpu_offload()
print("Pipeline loaded successfully.")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png"
)

processor = CannyDetector()
control_image = processor(
    control_image,
    low_threshold=50,
    high_threshold=200,
    detect_resolution=1024,
    image_resolution=1024,
)

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]

image.save(
    "Flux1-Canny-Dev-LoRA01-"
    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    + ".png"
)
