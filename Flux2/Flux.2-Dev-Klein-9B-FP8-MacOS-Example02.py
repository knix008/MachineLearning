import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os

# 로컬 safetensor 파일 경로
# HuggingFace에서 다운로드: https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8
# 파일명: flux-2-klein-9b-fp8.safetensors (9.43 GB)
model_path = os.path.join(os.path.dirname(__file__), "flux-2-klein-9b-fp8")

# 모델 디렉토리가 존재하는지 확인
if not os.path.exists(model_path):
    print(f"❌ 모델 디렉토리를 찾을 수 없습니다: {model_path}")
    print(f"✅ 다음 단계를 수행하세요:")
    print(f"   1. https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8 방문")
    print(f"   2. 'flux-2-klein-9b-fp8' 폴더와 파일 다운로드")
    print(f"   3. {model_path} 경로에 저장")
    exit(1)

print(f"✅ 모델 로드 중: {model_path}")

# 로컬 safetensor 모델 로드
pipe = Flux2KleinPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float8_e4m3fn,  # FP8 data type
    use_safetensors=True,
    local_files_only=True
)
#pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

device = "mps"

prompt = "Highly realistic, 4k, high-quality, high resolution, beautiful full body korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Perfect anatomy. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Orange hue, solid orange backdrop."

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0)
).images[0]

# Save with filename and datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"flux-klein_{timestamp}.png"
image.save(output_filename)
print(f"Image saved as: {output_filename}")