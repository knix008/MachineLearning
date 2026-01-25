import torch
from diffusers import Flux2Pipeline

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

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

image = pipe(
    prompt=prompt,
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=10, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")