import torch
import gc
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from datetime import datetime
import os
import sys

device = "cpu"
torch_dtype = torch.float32  # Use float32 for CPU

def cleanup_resources(pipe=None):
    """모든 자원을 해제하고 메모리를 정리합니다."""
    print("\n리소스 정리 중...")
    if pipe is not None:
        del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("리소스 정리 완료!")

# 시작 시 GPU 메모리 정리
print("GPU 메모리 초기화 중...")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
print("GPU 메모리 초기화 완료!")

pipe = None
try:
    repo_id = "black-forest-labs/FLUX.2-dev"  # Standard model

    # Load model on CPU with low memory usage
    pipe = Flux2Pipeline.from_pretrained(
        repo_id, torch_dtype=torch_dtype
    ).to(device)

    # Enable sequential CPU offload - moves layers to GPU only when needed
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()
    print("모델 로딩 완료!")

    prompt = "A highly realistic, high-quality photo of a beautiful Instagram-style korean girl on vacation. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, with a natural sparkle of happiness as she smiles. The image should capture her in a half-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."

    # Use the pipe directly - it handles text encoding internally
    # Generator device should match where the model's first layer is
    device_for_generator = "cpu"
    seed = 42
    guidance_scale = 4

    image = pipe(
        prompt=prompt,
        generator=torch.Generator(device=device_for_generator).manual_seed(seed),
        num_inference_steps=28,  # 28 steps can be a good trade-off
        guidance_scale=guidance_scale,
    ).images[0]

    # Save with program name and timestamp
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image.save(f"{script_name}_{timestamp}_seed{seed}_cfg{guidance_scale}.png")
    print(f"이미지 저장 완료: {script_name}_{timestamp}_seed{seed}_cfg{guidance_scale}.png")

except KeyboardInterrupt:
    print("\n\n사용자에 의해 중단되었습니다.")
    cleanup_resources(pipe)
    sys.exit(0)

except Exception as e:
    print(f"\n오류 발생: {e}")
    cleanup_resources(pipe)
    sys.exit(1)

finally:
    cleanup_resources(pipe)
