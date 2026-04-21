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

prompt = "A full body photography of a beautiful young skinny Korean woman standing on a casual spring outing in Seoul. She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera. She has long, voluminous wavy jet-black hair with beautiful soft waves and curls, dramatically flowing and billowing in the wind, strands sweeping through the air. Head held upright with elegant posture, hair draping naturally over shoulders. Standing perfectly still and upright, both feet and legs together, body facing completely straight toward the camera, chest and torso fully frontal, posture tall and elegant, shoulders back. One arm resting elegantly at her side, the other arm slightly bent with elbow relaxed, fashion model pose. One hand hanging gracefully at her side with fingers lightly extended, the other hand resting gently on her upper thigh with fingers elegantly spread, classic fashion model hand pose. Both legs naturally close together in a relaxed standing posture, side slit of skirt naturally parted revealing bare leg. Both feet naturally together, white sneakers clearly shown, feet not cropped. Dark navy chiffon one-piece dress with thin spaghetti straps, simple neckline, bare shoulders and arms, mostly opaque with only a slight translucency, densely scattered tiny cherry blossom print in soft pink and white, fitted waist, flowing A-line skirt with a side slit from the upper thigh naturally parting to reveal the bare leg, casual spring outing style. Bare legs, smooth and fair skin naturally visible through the flowing skirt slit. Clean white canvas sneakers, simple and casual. Bright spring street in Seoul, cherry blossom trees lining the sidewalk with pink petals falling gently, warm sunny day, clean pavement. Bright even spring daylight, soft frontal natural light, face clearly and brightly lit, no harsh shadows. Full body shot, entire body from head to feet fully in frame, feet and sneakers not cropped, slightly low angle to elongate legs, subject facing camera, sharp focus, soft bokeh background. 8k, high quality, realistic, detailed, sharp focus, perfect anatomy, ten fingers."

NEGATIVE_PROMPT = "Blurry, low quality, deformed, bad anatomy, extra limbs, ugly, watermark, text, signature, extra fingers, one leg forward, staggered legs, walking pose, weight shift, legs apart, stepping."

width, height = aspect_ratios["9:16"]

image = pipe(
    prompt=prompt,
    negative_prompt=NEGATIVE_PROMPT,
    width=width,
    height=height,
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("nucleus_output.png")
