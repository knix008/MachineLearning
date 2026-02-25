import torch
import os
import time
import platform
import psutil
from datetime import datetime
from diffusers import Flux2KleinPipeline

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

# Hardware info
cpu_name = platform.processor() or platform.machine()
ram_gb = psutil.virtual_memory().total / (1024 ** 3)
print(f"CPU  : {cpu_name}")
print(f"RAM  : {ram_gb:.1f} GB")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {vram_gb:.1f} GB")
elif device == "mps":
    print(f"GPU  : Apple Silicon (MPS)")
    try:
        import subprocess
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True
        )
        import json
        displays = json.loads(result.stdout).get("SPDisplaysDataType", [{}])
        vram_info = displays[0].get("sppci_memory", "N/A") if displays else "N/A"
        print(f"VRAM : {vram_info} (unified)")
    except Exception:
        print(f"VRAM : N/A")
else:
    print(f"GPU  : None")
    print(f"VRAM : N/A")
print()

# Memory optimization per device
# cuda: sequential offload > model offload > attention slicing
# mps : pipe.to(device) only
# cpu : sequential offload > model offload > attention slicing
memory_opts = {
    "cuda": {"sequential_offload": True, "model_cpu_offload": True, "attention_slicing": True},
    "mps":  {"sequential_offload": False, "model_cpu_offload": False, "attention_slicing": False},
    "cpu":  {"sequential_offload": True, "model_cpu_offload": True, "attention_slicing": True},
}
opts = memory_opts[device]

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype)
if device == "mps":
    pipe = pipe.to(device)
else:
    if opts["sequential_offload"]:
        pipe.enable_sequential_cpu_offload()
    elif opts["model_cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if opts["attention_slicing"]:
        pipe.enable_attention_slicing()

print(f"Memory opts: {opts}")

prompt = "The image is a high-quality, photorealistic portrait of a young Korean woman with a soft, idol aesthetic. She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera. She has long, voluminous wavy jet-black hair with beautiful soft waves and curls, dramatically flowing and billowing in the wind, strands sweeping through the air with natural movement and body, full of life and dynamism. She is wearing an extremely tiny black lingerie set, barely covering her body. A very small black bra and matching micro black panties, delicate and sensual, the soft fabric clinging gently to her skin. Full body shot, walking gracefully along the beach shoreline toward the camera. Natural and relaxed walking gait, one foot stepping forward. Both arms hanging naturally down at her sides, swinging loosely with the natural rhythm of walking. Head facing forward toward the camera with a confident and alluring expression. Hair flowing freely in the sea breeze. Luxurious resort beach with white sand shoreline. Modern high-rise resort towers in the background skyline. Ocean waves at the shore. Bright natural sunlight, golden hour warm tones. Soft warm highlights on her skin. Sparkling ocean water in the background. Cinematic warm beach lighting. Vertical full body portrait, chest-level shot. 85mm portrait lens, shallow depth of field, resort buildings and ocean softly blurred. Realistic lifestyle beach photography. Ultra-realistic masterpiece photograph, 8k resolution, high-fidelity skin textures, cinematic lighting, realistic lifestyle photography, photorealistic, sharp focus. Perfect anatomy, correct finger count, no deformed or fused fingers, perfect hand structure, perfect feet structure, perfect body proportion, no extra hands, no extra feet,no distorted body."

height = 1024
width = 1024
guidance_scale = 1.0
num_inference_steps = 4
seed = 0

start_time = time.time()
image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator(device=device).manual_seed(seed)
).images[0]
elapsed = time.time() - start_time
print(f"\nInference time: {elapsed:.1f}s")

script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"{script_name}_{timestamp}_{device.upper()}_{width}x{height}_gs{guidance_scale}_step{num_inference_steps}_seed{seed}.png"
image.save(output_filename)
print(f"Saved: {output_filename}")