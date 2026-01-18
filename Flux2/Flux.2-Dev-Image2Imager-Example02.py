import torch
from diffusers import FluxImg2ImgPipeline
from datetime import datetime
from PIL import Image

device = "cpu"
dtype = torch.float32

# Load image-to-image pipeline (FLUX.1-dev supports img2img, FLUX.2-klein does not)
pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=dtype
)
pipe = pipe.to(device)

pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
pipe.enable_attention_slicing(1)  # reduce memory usage further
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")

# Load input image
input_image_path = "default.jpg"  # Change this to your image path
input_image = Image.open(input_image_path).convert("RGB")
print(f"이미지 로드 완료: {input_image_path}")

prompt = "8k resolution quality, high detail, high quality, best quality, realistic, masterpiece, dark blue bikini. The image should capture her in a full-body shot, with perfect anatomy, avoiding an overly smooth or filtered look, to maintain a lifelike. The overall atmosphere is bright, reflecting the sunny, relaxed vacation mood."
image = pipe(
    prompt=prompt,
    image=input_image,
    width=input_image.width,
    height=input_image.height,
    strength=0.85,  # Adjust this value (0.0-1.0) to control how much the input image should be changed
    guidance_scale=3.5,
    num_inference_steps=28,
    generator=torch.Generator(device=device).manual_seed(100),
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"flux-dev-img2img_{timestamp}.png"
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")
