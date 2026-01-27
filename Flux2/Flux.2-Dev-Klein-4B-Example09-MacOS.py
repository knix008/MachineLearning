import torch
import os
from diffusers import Flux2KleinPipeline
from datetime import datetime

device = "mps"
dtype = torch.float16

print("모델을 로딩하는 중입니다...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype
)
pipe = pipe.to(device)
print("모델 로딩 완료!")

prompt = "A highly realistic, 4k resolution, high-quality photo of a beautiful skinny Instagram-style korean girl model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. Her eyes are hazel, wearing a red bikini, with a natural sparkle of happiness as she smiles. The image should be perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike. Orange hue, solid orange backdrop,using a camera setup that mimics a large aperture,f/1.4 --ar 9:16 --style raw."

print("이미지를 생성하는 중입니다...")

image = pipe(
    prompt=prompt,
    height=1024,
    width=512,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_name = os.path.splitext(os.path.basename(__file__))[0]
filename = f"{script_name}_{timestamp}.png"
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")