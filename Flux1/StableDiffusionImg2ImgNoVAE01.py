import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V6.0_B1_noVAE", torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("default.jpg").convert("RGB").resize((768, 768))
prompt = "a realistic portrait, 8k, high detail, masterpiece"

result = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.7,  # 0.3~0.8 추천, 높을수록 원본 변화 큼
    guidance_scale=7.0,
    num_inference_steps=30,
).images[0]

result.save("realistic_img2img_result.png")