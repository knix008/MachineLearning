import torch
import os
import time
import platform
import psutil
from datetime import datetime
from tqdm import tqdm
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
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"CPU  : {cpu_name}")
print(f"RAM  : {ram_gb:.1f} GB")

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {vram_gb:.1f} GB")
elif device == "mps":
    print(f"GPU  : Apple Silicon (MPS)")
    try:
        import subprocess

        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
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
# mps : attention slicing only
# cpu : sequential offload > model offload > attention slicing
memory_opts = {
    "cuda": {
        "sequential_offload": True,
        "model_cpu_offload": True,
        "attention_slicing": True,
    },
    "mps": {
        "sequential_offload": False,
        "model_cpu_offload": False,
        "attention_slicing": True,
    },
    "cpu": {
        "sequential_offload": True,
        "model_cpu_offload": True,
        "attention_slicing": True,
    },
}
opts = memory_opts[device]

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype
)
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

# Subject
p_subject = "The image is a high-quality, photorealistic portrait of a young Korean woman with a soft, idol aesthetic."

# Face & Appearance
p_face = (
    "She has a fair, clear complexion. "
    "She is wearing striking bright blue contact lenses that contrast with her dark hair. "
    "Her expression is innocent and curious, looking directly at the camera."
)

# Hair
p_hair = (
    "She has long, voluminous wavy jet-black hair with beautiful soft waves and curls, "
    "dramatically flowing and billowing in the wind, strands sweeping through the air "
    "with natural movement and body, full of life and dynamism. "
    "Hair flowing freely in the sea breeze."
)

# Outfit
p_outfit = (
    "She is wearing an extremely tiny black lingerie set, barely covering her body. "
    "A very small black bra and matching micro black panties, delicate and sensual, "
    "the soft fabric clinging gently to her skin."
)

# Pose & Action
p_pose = (
    "Full body shot, walking gracefully along the beach shoreline toward the camera. "
    "Natural and relaxed walking gait, one foot stepping forward. "
    "Both arms hanging naturally down at her sides, swinging loosely with the natural rhythm of walking. "
    "Head facing forward toward the camera with a confident and alluring expression."
)

# Background & Setting
p_background = (
    "Luxurious resort beach with white sand shoreline. "
    "Modern high-rise resort towers in the background skyline. "
    "Ocean waves at the shore. "
    "Sparkling ocean water in the background."
)

# Lighting
p_lighting = (
    "Bright natural sunlight, golden hour warm tones. "
    "Soft warm highlights on her skin. "
    "Cinematic warm beach lighting."
)

# Camera & Shot Style
p_camera = (
    "Vertical full body portrait, chest-level shot. "
    "85mm portrait lens, shallow depth of field, resort buildings and ocean softly blurred. "
    "Realistic lifestyle beach photography."
)

# Quality & Technical
p_quality = (
    "Ultra-realistic masterpiece photograph, 8k resolution, high-fidelity skin textures, "
    "cinematic lighting, realistic lifestyle photography, photorealistic, sharp focus."
)

# Anatomy
p_anatomy = (
    "Perfect anatomy, correct finger count, no deformed or fused fingers, "
    "perfect hand structure, perfect feet structure, perfect body proportion, "
    "no extra hands, no extra feet, no distorted body."
)

prompt = " ".join(
    [
        p_subject,
        p_face,
        p_hair,
        p_outfit,
        p_pose,
        p_background,
        p_lighting,
        p_camera,
        p_quality,
        p_anatomy,
    ]
)

# Check prompt truncation
max_sequence_length = 512
tokens = pipe.tokenizer(prompt, return_tensors="pt")
token_count = tokens.input_ids.shape[1]
if token_count > max_sequence_length:
    truncated_text = pipe.tokenizer.decode(
        tokens.input_ids[0, max_sequence_length:], skip_special_tokens=True
    )
    print(
        f"WARNING: Prompt truncated! {token_count} tokens > {max_sequence_length} max."
    )
    print(f"Truncated text: '{truncated_text}'")
else:
    print(f"Prompt tokens: {token_count}/{max_sequence_length}")

height = 1024
width = 1024
guidance_scale = 1.0
num_inference_steps = 4
seed = 0

pbar = tqdm(
    total=num_inference_steps,
    desc="Inference",
    unit="step",
    dynamic_ncols=True,
    bar_format=(
        "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} "
        "[elapsed: {elapsed} | remaining: {remaining} | {rate_fmt}]"
    ),
)

start_time = time.time()
last_step_time = [start_time]


def step_callback(pipe, step, timestep, callback_kwargs):
    now = time.time()
    step_time = now - last_step_time[0]
    last_step_time[0] = now
    completed = step + 1
    avg = (now - start_time) / completed
    est_total = avg * num_inference_steps
    pbar.set_postfix(
        {
            "s/step": f"{step_time:.1f}s",
            "est_total": f"{est_total:.0f}s",
        },
        refresh=False,
    )
    pbar.update(1)
    return callback_kwargs


image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator(device=device).manual_seed(seed),
    callback_on_step_end=step_callback,
).images[0]
pbar.close()
elapsed = time.time() - start_time
avg_step = elapsed / num_inference_steps
print(f"Inference time: {elapsed:.1f}s  |  avg: {avg_step:.2f}s/step")

script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"{script_name}_{timestamp}_{device.upper()}_{width}x{height}_gs{guidance_scale}_step{num_inference_steps}_seed{seed}.png"
image.save(output_filename)
print(f"Saved: {output_filename}")
