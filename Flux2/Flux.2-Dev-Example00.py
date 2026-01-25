import torch
from diffusers import Flux2Pipeline
from datetime import datetime
import os

# 일반 FLUX.2-dev 모델 사용 (양자화 모델에 문제가 있음)
repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

pipe = Flux2Pipeline.from_pretrained(
    repo_id, 
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,  # CPU 메모리 사용량 감소
)

if torch.cuda.is_available():
    # CPU offload를 사용하여 GPU 메모리 부담 줄이기
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()  # 순차적 CPU offload로 메모리 최적화
    pipe.vae.enable_tiling()  # VAE 타일링으로 메모리 사용량 감소
    pipe.vae.enable_slicing()  # VAE 슬라이싱 활성화
else:
    pipe = pipe.to(device)
print("모델 로딩 완료!")

prompt = "Highly realistic, 4k, high-quality, high resolution, beautiful korean woman model photography. having black medium-length hair reaching her shoulders, tied back, wearing a red bikini, looking at the viewer. Perfect anatomy, solid orange backdrop, using a camera setup that mimics a large aperture f/1.4, ar 9:16, style raw.s"

image = pipe(
    prompt=prompt,
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=10, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

# Generate output filename with script name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{script_name}_{timestamp}.png"

image.save(output_filename)
print(f"Image saved as: {output_filename}")