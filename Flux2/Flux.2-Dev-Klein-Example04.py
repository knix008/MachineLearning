import torch
from diffusers import Flux2KleinPipeline
from datetime import datetime

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=dtype
)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

prompt = "A highly realistic, high-quality photo of a beautiful skinny Instagram-style girl on vacation. She has blonde, medium-length hair that reaches her shoulders, tied back in a casual yet stylish manner. She is walking on a sunny beach and her eyes are hazel, wearing a red bikini, with a natural sparkle of happiness as she smiles. The image should capture her in a full-body shot, with perfect anatomy, including precise details in her eyes and teeth. Her skin should appear natural, with visible pores, avoiding an overly smooth or filtered look, to maintain a lifelike, 4K resolution quality. The overall atmosphere is bright and joyful, reflecting the sunny, relaxed vacation mood."
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(100),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"flux-klein_{timestamp}.png"
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")
