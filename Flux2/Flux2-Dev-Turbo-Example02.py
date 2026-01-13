import torch
from diffusers import Flux2Pipeline

# https://www.aitimes.com/news/articleView.html?idxno=205183 to get the access right.

# from huggingface_hub import login
#
# access_token = "hf_"
# login(access_token)

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
print("모델 로딩 완료!")

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)

prompt = "a skinny body woman having beautiful smile on her face is walking on a sunset beach, wearing a red bikini, cinematic lighting, detailed, 8k"

image = pipe(
    prompt=prompt,
    sigmas=TURBO_SIGMAS,
    guidance_scale=2.5,
    height=1024,
    width=1024,
    num_inference_steps=8,
).images[0]

image.save("output.png")
