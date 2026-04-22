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

prompt = "A weathered lighthouse on a rocky coastline at golden hour, waves crashing against the rocks below, seagulls circling overhead, dramatic clouds painted in shades of amber and violet"
width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("nucleus_output.png")
