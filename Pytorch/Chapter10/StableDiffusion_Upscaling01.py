import torch
from diffusers import StableDiffusionPipeline, StableDiffusionLatentUpscalePipeline

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 기본 SD 2.0 모델 로드
base_model = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16, variant="fp16"
).to(device)

# 프롬프트 설정
prompt = "a vivid blue and yellow macaw in a tropical jungle, cinematic lighting, ultra-high detail"

# 중간 해상도 이미지 생성
low_res_image = pipe(prompt=prompt, height=768, width=1152).images[0]

# 2. 업스케일 모델 로드
upscale_model = "stabilityai/stable-diffusion-x4-upscaler"
upscale_pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
    upscale_model, torch_dtype=torch.float16
).to(device)

# 고해상도 업스케일 이미지 생성
high_res_image = upscale_pipe(
    prompt=prompt, image=low_res_image, num_inference_steps=50, guidance_scale=7.5
).images[0]

# 저장
high_res_image.save("macaw_highres_upscaled.png")
print("업스케일된 이미지가 저장되었습니다.")
