import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image
import datetime

# --- 1. 모델 및 파이프라인 설정 ---
# GPU 사용 설정 (사용 가능한 경우)
device = "cuda" if torch.cuda.is_available() else "cpu"
# 메모리 효율을 위해 bfloat16 데이터 타입 사용
torch_dtype = torch.bfloat16

print(f"Using device: {device}")

# FLUX.1-Kontext-dev 기본 파이프라인 불러오기
# 처음 실행 시 모델 파일을 자동으로 다운로드합니다.
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch_dtype
)

# --- 2. LoRA(로라) 모델 불러오기 및 적용 ---
# "make-person-real" LoRA 가중치를 불러옵니다.
# 로컬에 저장된 .safetensors 파일 경로를 직접 지정할 수도 있습니다.
# 예: lora_path = "./models/loras/flux-kontext-make-person-real-lora.safetensors"
lora_path = "flux-kontext-make-person-real-lora.safetensors"
pipe.load_lora_weights(lora_path, prefix=None)

# 파이프라인을 GPU로 이동
pipe.to(device)

print("Model and LoRA loaded successfully.")

# --- 3. 입력값 준비 ---
# 변환할 이미지 불러오기
# 이미지 URL 또는 로컬 파일 경로를 사용하세요.
image_path = "default.jpg"
def load_image(image_path):
    """이미지를 로드하고 RGB로 변환하며, 한 변이 1024를 넘지 않도록 비율 유지 리사이즈."""
    image = Image.open(image_path).convert("RGB")
    max_size = 1024
    w, h = image.size
    if w > max_size or h > max_size:
        scale = min(max_size / w, max_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image
input_image = load_image(image_path)

# 프롬프트 설정
prompt = "make this person look real, 8k uhd, high detail skin texture, detailed eyes"

# --- 4. 이미지 생성 실행 ---
# 파이프라인을 실행하여 이미지 생성
# guidance_scale 값을 조절하여 프롬프트의 영향력을 제어할 수 있습니다.
result_image = pipe(
    prompt=prompt,
    image=input_image,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

print("Image generation complete.")

# --- 5. 결과 저장 ---
# 결과 이미지를 파일로 저장
output_path = f"result_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
result_image.save(output_path)

print(f"Image saved to {output_path}")