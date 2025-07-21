import torch
from diffusers import DiffusionPipeline
from PIL import Image
import gc  # Garbage Collector

# --- 1. 필수 라이브러리 설치 ---
# !pip install diffusers transformers accelerate safetensors torch xformers

# --- 2. 모델 로딩 및 메모리 최적화 설정 ---

# 사용할 모델 ID (빠른 버전인 schnell)
model_id = "black-forest-labs/FLUX.1-schnell"

# 파이프라인 로딩 시 torch.float16을 사용하여 메모리 사용량을 절반으로 줄입니다.
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # bfloat16을 사용하여 메모리 사용량을 줄입니다.
    #torch_dtype=torch.float16,
    #variant="fp16",  # fp16 버전을 명시적으로 사용
)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
pipe.enable_vae_slicing()  # Slice VAE computation
print("모델 로딩 완료!")

# ⭐ 핵심 최적화 2: xformers를 사용한 메모리 효율적 어텐션 활성화
# 어텐션 계산 시 메모리 사용량을 크게 줄여줍니다. xformers 라이브러리가 설치되어 있어야 합니다.
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers 메모리 효율적 어텐션 활성화 완료.")
except (ImportError, AttributeError) as e:
    print(f"xformers 어텐션 최적화를 건너뜁니다: {e}")
    # Alternative: Use torch's built-in attention optimization
    try:
        pipe.unet.set_attn_processor(None)  # Use default attention
        print("기본 어텐션 프로세서를 사용합니다.")
    except Exception:
        print("어텐션 최적화를 건너뜁니다.")


# (선택) 추가 최적화: VAE Tiling
# VAE(이미지 인코더/디코더)가 메모리를 많이 사용할 경우, 이미지를 타일 단위로 처리해 메모리를 절약합니다.
# 고해상도 이미지 생성 시 OutOfMemory 에러가 발생하면 활성화하세요.
# pipe.enable_vae_tiling()


# --- 3. 이미지 생성 (Inference) ---
prompt = "photo of a beautiful golden retriever puppy playing in a field of yellow flowers, cinematic lighting, ultra-detailed"
negative_prompt = "blurry, low quality, cartoon, ugly"

# torch.no_grad() 컨텍스트 안에서 실행하여 불필요한 그래디언트 계산을 방지합니다.
with torch.no_grad():
    print("\n이미지 생성을 시작합니다...")
    # 생성 파라미터 설정
    generated_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
    ).images[0]
    print("이미지 생성 완료!")

# --- 4. 결과 저장 및 메모리 정리 ---

# 생성된 이미지 저장
output_path = "flux1_optimized_output.png"
generated_image.save(output_path)
print(f"생성된 이미지를 '{output_path}'에 저장했습니다.")

# 파이프라인과 캐시를 정리하여 VRAM을 확보합니다.
del pipe
gc.collect()
torch.cuda.empty_cache()
print("GPU 캐시 및 메모리를 정리했습니다.")
