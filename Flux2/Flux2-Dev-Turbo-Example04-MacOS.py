import torch
import os
import gc
from diffusers import Flux2Pipeline
from datetime import datetime

# MPS 메모리 최적화 - 프로그램 시작 시 설정
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Apple Silicon 설정
device = "mps"
dtype = torch.float16  # MPS는 float16 지원이 더 안정적

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

print("모델 로딩 중...")
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=dtype
).to(device)

# Enable memory optimizations for Apple Silicon
pipe.enable_attention_slicing(1)
pipe.enable_model_cpu_offload()
print(f"모델 로딩 완료! (Device: {device}, Dtype: {dtype})")

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)

prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

torch.mps.empty_cache()
gc.collect()

image = pipe(
    prompt=prompt,
    sigmas=TURBO_SIGMAS,
    guidance_scale=2.5,
    height=1024,
    width=1024,
    num_inference_steps=8,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

# Generate filename with script name and current date/time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_name = os.path.splitext(os.path.basename(__file__))[0]
output_filename = f"{script_name}_{timestamp}.png"

image.save(output_filename)
print(f"이미지가 저장되었습니다: {output_filename}")

# 메모리 정리
torch.mps.empty_cache()
gc.collect()
