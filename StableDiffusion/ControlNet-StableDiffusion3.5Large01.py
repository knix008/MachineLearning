import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image

# 1. 제어 이미지 준비 및 전처리
# 제어에 사용할 이미지를 불러옵니다.
control_image_url = (
    "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
)
original_image = load_image(control_image_url)

# OpenCV를 사용하여 Canny Edge를 추출합니다.
image_np = np.array(original_image)
canny_image = cv2.Canny(image_np, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
control_image = Image.fromarray(canny_image)

# 2. ControlNet 및 Stable Diffusion 3.5 모델 불러오기
# Hugging Face Hub에서 사전 학습된 ControlNet 모델과 SD 3.5 모델을 불러옵니다.
# 메모리 효율을 위해 float16 정밀도를 사용합니다.
controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

# GPU로 모델을 이동시켜 빠른 연산을 수행합니다.
pipe.to("cuda")

# 3. 이미지 생성
# 프롬프트를 정의하고 파이프라인을 실행하여 이미지를 생성합니다.
prompt = "A beautiful photo of a majestic parrot on a branch, high quality, detailed."

# `control_image`가 제어 신호로, `prompt`가 내용 서술로 사용됩니다.
generated_image = pipe(
    prompt,
    control_image=control_image,
    num_inference_steps=25,
    guidance_scale=7.5,
).images[0]

# 4. 결과 저장
# 생성된 이미지를 파일로 저장합니다.
generated_image.save("parrot_with_controlnet.png")

print("이미지 생성이 완료되었습니다: parrot_with_controlnet.png")
