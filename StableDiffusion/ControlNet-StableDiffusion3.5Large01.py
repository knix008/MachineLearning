import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np
from PIL import Image

# 1. 제어 이미지 준비 (Canny Edge 추출) 🖼️
# 제어로 사용할 원본 이미지를 불러옵니다.
url = (
    "https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/images/bird.png"
)
original_image = load_image(url)

# OpenCV를 사용하여 Canny Edge를 검출합니다.
image_np = np.array(original_image)
canny_image = cv2.Canny(image_np, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
control_image = Image.fromarray(canny_image)

# 2. 모델 로딩 🧠
# 사용할 ControlNet 모델과 Stable Diffusion 3.5 베이스 모델을 불러옵니다.
# 'canny' 외에 'depth', 'pose' 등 다른 ControlNet을 사용하려면 모델 경로를 변경하면 됩니다.
controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

# GPU (CUDA)로 모델을 이동시켜 연산 속도를 높입니다.
pipe.to("cuda")

# 3. 이미지 생성 🎨
# 텍스트 프롬프트를 정의합니다.
prompt = """A beautiful photo of a majestic parrot on a branch, 
photorealistic, ultra high definition, 8k resolution, ultra detail, 
vibrant colors, cinematic lighting, realistic shadows, high quality, 
masterpiece, best quality, perfect anatomy"""

# 네거티브 프롬프트 (생성하지 않을 요소들)
negative_prompt = """blurry, low quality, bad anatomy, bad hands, text, error, 
missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, 
username, deformed, distorted, disfigured, mutation, mutated"""

# 파이프라인을 실행하여 이미지를 생성합니다.
# control_image가 구조를, prompt가 내용을 결정합니다.
generated_image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    control_image=control_image,
    num_inference_steps=25,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,
).images[0]

# 4. 결과 저장 💾
generated_image.save("parrot_generated_with_canny_control.png")
print("이미지 생성이 완료되었습니다: parrot_generated_with_canny_control.png")
