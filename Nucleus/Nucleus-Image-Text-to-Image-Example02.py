import torch
from diffusers import DiffusionPipeline
from diffusers import TextKVCacheConfig

model_name = "NucleusAI/Nucleus-Image"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)
pipe.to("cpu")

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

# Enable Text KV caching across denoising steps (integrated into diffusers)
config = TextKVCacheConfig()
pipe.transformer.enable_cache(config)

# Supported aspect ratios
aspect_ratios = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1184, 896),
    "3:4": (896, 1184),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
}

prompt = "Editorial full-body portrait photo of a young Korean woman on a spring street in Seoul. Subject centered, standing still, front-facing to camera, relaxed natural posture with both feet visible and not cropped. Neutral gentle expression, looking directly at the camera. Long jet-black wavy hair moving softly in a light breeze. Outfit: dark navy chiffon one-piece dress with tiny pink-white cherry blossom print, fitted waist, flowing A-line skirt with a subtle side slit, clean white canvas sneakers. Environment: bright warm spring daylight, cherry blossom trees along the sidewalk, a few falling petals, clean urban street. Camera/style: full-body framing from head to shoes, slightly low angle, 85mm photo lens look, realistic skin texture, sharp subject focus, soft background bokeh, natural color grading, highly detailed, photorealistic, 8k."

NEGATIVE_PROMPT = "blurry, low quality, out of focus, deformed anatomy, bad hands, extra fingers, missing fingers, extra limbs, duplicate body, cropped feet, cropped shoes, partial body, twisted pose, walking pose, crossed legs, legs apart, text, logo, watermark, signature, oversaturated colors, harsh shadows"

width, height = aspect_ratios["9:16"]

# Recommended baseline for Nucleus image generation:
# - Lower guidance keeps anatomy/composition natural.
# - Mid-step sampling balances detail and stability.
num_inference_steps = 36
guidance_scale = 3.5

image = pipe(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("nucleus_output.png")
