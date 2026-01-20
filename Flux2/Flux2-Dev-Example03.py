import torch
from diffusers import FluxPipeline
from datetime import datetime
from PIL import Image

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
prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look."

# Run the pipeline
image = pipe(
    prompt=prompt,
    width=512,
    height=1024,
    guidance_scale=2.5,
    num_inference_steps=28,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

# Save with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"Flux2-Dev-Example03_{timestamp}.png"
# Save the generated image
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")
