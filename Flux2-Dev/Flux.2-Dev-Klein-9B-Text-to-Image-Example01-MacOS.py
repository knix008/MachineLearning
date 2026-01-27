import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime
import os

device = "mps"
dtype = torch.bfloat16

print("모델 로딩 중... 기다려주세요.")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype
)
pipe = pipe.to(device)

#pipe.enable_model_cpu_offload()
#pipe.enable_attention_slicing()
#pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")


prompt = "Highly realistic, 4k, high-quality, high resolution, beautiful full body korean woman model photography. She has black, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner, wearing a red bikini. Her eyes are hazel, with a natural sparkle of happiness as she smiles. Her skin appears natural with visible pores. Orange hue, solid orange backdrop, using a camera setup that mimics a large aperture, f/1.4 --ar 9:16 --style raw."

image = pipe(
    prompt=prompt,
    height=1024,
    width=512,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{script_name}_{timestamp}.png"
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")
