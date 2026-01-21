import torch
from diffusers import FluxPipeline
from datetime import datetime
from PIL import Image
import os

# Set device and data type
device = "cpu"
dtype = torch.float32

# Load text-to-image pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
).to(device)

# Enable memory optimizations
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.enable_attention_slicing(1)  # reduce memory usage further
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")

# Set the prompt. It should be less than 77 words for best results.
prompt = "A highly realistic, high-quality photo of a beautiful skinny Instagram-style girl on vacation. She has blonde, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, wearing a red bikini, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

# Run the pipeline
image = pipe(
    prompt=prompt,
    width=720,
    height=1024,
    guidance_scale=4.0,
    num_inference_steps=28,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

# Save with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Get the current script filename without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]
filename = f"{script_name}_{timestamp}.png"
# Save the generated image
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")
