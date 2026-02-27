import torch
import os
import time
import platform
import psutil
import subprocess
import json
from datetime import datetime
import gradio as gr
from tqdm import tqdm
from diffusers import FluxPipeline

# ── Device detection ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

# ── Hardware info ─────────────────────────────────────────────────────────────
cpu_name = platform.processor() or platform.machine()
ram_gb = psutil.virtual_memory().total / (1024**3)

if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    hw_info = f"CPU: {cpu_name} | RAM: {ram_gb:.1f} GB | GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB"
elif device == "mps":
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
        )
        displays = json.loads(result.stdout).get("SPDisplaysDataType", [{}])
        vram_info = displays[0].get("sppci_memory", "N/A") if displays else "N/A"
    except Exception:
        vram_info = "N/A"
    hw_info = f"CPU: {cpu_name} | RAM: {ram_gb:.1f} GB | GPU: Apple Silicon (MPS) | VRAM: {vram_info} (unified)"
else:
    hw_info = f"CPU: {cpu_name} | RAM: {ram_gb:.1f} GB | GPU: None"

print(hw_info)

# ── Memory optimization ───────────────────────────────────────────────────────
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

# ── Load pipeline ─────────────────────────────────────────────────────────────
print("Loading pipeline...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    tokenizer=None,
    torch_dtype=dtype,
)
if device == "mps":
    pipe = pipe.to(device)
    if hasattr(pipe, "transformer"):
        pipe.transformer.to(memory_format=torch.channels_last)
else:
    if opts["sequential_offload"]:
        pipe.enable_sequential_cpu_offload()
    elif opts["model_cpu_offload"]:
        pipe.enable_model_cpu_offload()
if opts["attention_slicing"]:
    pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)  # 내장 tqdm 비활성화 - 커스텀 진행 바 사용
print(f"Pipeline ready. Device: {device.upper()} | Memory opts: {opts}")

# ── Default prompt sections ───────────────────────────────────────────────────

# Subject
DEFAULT_SUBJECT = "The image is a high-quality, photorealistic portrait of a young Korean woman with a soft, idol aesthetic."

# Face & Appearance
DEFAULT_FACE = "She has a fair, clear complexion. She is wearing striking bright blue contact lenses that contrast with her dark hair. Her expression is innocent and curious, looking directly at the camera."

# Hair
DEFAULT_HAIR = "She has long, voluminous wavy jet-black hair with beautiful soft waves and curls, dramatically flowing and billowing in the wind, strands sweeping through the air with natural movement and body, full of life and dynamism. Hair flowing freely in the sea breeze."

# Outfit
DEFAULT_OUTFIT = "She is wearing a light blue bikini swimsuit. The top is a small triangle bikini top with thin spaghetti straps tied behind the neck and back, snugly fitting her chest. The bottom is a tiny light blue bikini bottom with thin side-tie strings resting on her hips, high-cut on the sides. The soft pastel blue fabric contrasts delicately against her fair skin. No cover-up or additional clothing."

# Pose & Action
DEFAULT_POSE = "She is walking gracefully along the beach shoreline toward the camera. Natural and relaxed walking gait, one foot stepping forward. Both arms hanging naturally down at her sides, swinging loosely with the natural rhythm of walking. Head facing forward toward the camera with a confident and alluring expression."

# Background & Setting
DEFAULT_BACKGROUND = "Luxurious resort beach with white sand shoreline. Modern high-rise resort towers in the background skyline. Ocean waves at the shore. Sparkling ocean water in the background."

# Lighting
DEFAULT_LIGHTING = "Bright natural sunlight, golden hour warm tones. Soft warm highlights on her skin. Cinematic warm beach lighting."

# Camera & Shot Style
DEFAULT_CAMERA = "Vertical full body portrait, chest-level shot. 85mm portrait lens, shallow depth of field, resort buildings and ocean softly blurred. Realistic lifestyle beach photography."

# Quality & Technical
DEFAULT_QUALITY = "Ultra-realistic masterpiece photograph, 8k resolution, high-fidelity skin textures, cinematic lighting, realistic lifestyle photography, photorealistic, sharp focus."

# Anatomy
DEFAULT_ANATOMY = "Perfect anatomy, correct finger count, no deformed or fused fingers, perfect hand structure, perfect feet structure, perfect body proportion, no extra hands, no extra feet, no distorted body."


# ── Generate function ─────────────────────────────────────────────────────────
def generate(*args, progress=gr.Progress()):
    (
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
        height,
        width,
        guidance_scale,
        num_inference_steps,
        seed,
        max_sequence_length,
        image_format,
    ) = args

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

    # Check prompt truncation (T5-XXL tokenizer)
    max_len = int(max_sequence_length)
    raw_ids = pipe.tokenizer_2(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
    raw_token_count = len(raw_ids)
    clipped = max(0, raw_token_count - max_len)
    if clipped > 0:
        truncated_text = pipe.tokenizer_2.decode(raw_ids[max_len:], skip_special_tokens=True)
        print(f"WARNING: Prompt truncated! {raw_token_count} tokens > {max_len} max.")
        print(f"Truncated text: '{truncated_text}'")
    else:
        print(f"Prompt tokens: {raw_token_count}/{max_len}")

    steps = int(num_inference_steps)
    progress(0, desc="Starting...")

    pbar = tqdm(
        total=steps,
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

    def step_callback(_pipe, step, _timestep, callback_kwargs):
        now = time.time()
        step_time = now - last_step_time[0]
        last_step_time[0] = now
        completed = step + 1
        avg = (now - start_time) / completed
        est_total = avg * steps
        est_remaining = est_total - (now - start_time)
        pbar.set_postfix(
            {
                "s/step": f"{step_time:.1f}s",
                "est_total": f"{est_total:.0f}s",
            },
            refresh=False,
        )
        pbar.update(1)
        desc = (
            f"Step {completed}/{steps} | "
            f"elapsed: {now - start_time:.1f}s | "
            f"remaining: {max(est_remaining, 0):.1f}s | "
            f"est_total: {est_total:.1f}s | "
            f"s/step: {step_time:.1f}s"
        )
        progress(completed / steps, desc=desc)
        return callback_kwargs

    # Encode prompt using T5-XXL only (CLIP disabled)
    generator_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))

    text_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        prompt_embeds = pipe.text_encoder_2(
            text_inputs["input_ids"].to(device),
            output_hidden_states=False,
        )[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    pooled_prompt_embeds = torch.zeros(1, 768, dtype=dtype, device=prompt_embeds.device)

    with torch.inference_mode():
        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            height=int(height),
            width=int(width),
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
            callback_on_step_end=step_callback,
        ).images[0]
    pbar.close()
    elapsed = time.time() - start_time
    avg_step = elapsed / steps
    progress(1, desc=f"Done | total: {elapsed:.1f}s | avg: {avg_step:.2f}s/step")

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ext = "jpg" if image_format == "JPEG" else "png"
    output_filename = (
        f"{script_name}_{timestamp}_{device.upper()}_"
        f"{int(width)}x{int(height)}_gs{guidance_scale}_step{int(num_inference_steps)}_seed{int(seed)}"
        f"_msl{int(max_sequence_length)}.{ext}"
    )
    if image_format == "JPEG":
        image.save(output_filename, format="JPEG", quality=100, subsampling=0)
    else:
        image.save(output_filename)

    token_info = (
        f"Tokens: {raw_token_count}/{max_len} → {clipped} truncated!"
        if clipped > 0
        else f"Tokens: {raw_token_count}/{max_len}"
    )
    info = f"Inference time: {elapsed:.1f}s | {token_info} | Saved: {output_filename}"
    print(info)
    return image, info


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="FLUX.1-dev Text-to-Image") as demo:
    gr.Markdown("# FLUX.1-dev Text-to-Image")
    gr.Markdown(f"**Hardware:** {hw_info}")
    gr.Markdown(f"**Device:** `{device.upper()}` | **Memory opts:** `{opts}`")

    with gr.Row():
        # ── Left: Prompt Sections ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Prompt Sections")
            p_subject = gr.Textbox(
                label="Subject",
                value=DEFAULT_SUBJECT,
                lines=2,
                info="주인공의 전반적인 설명. 인물 유형, 나이, 분위기, 스타일 등. 예: 'photorealistic portrait of a young Korean woman'",
            )
            p_face = gr.Textbox(
                label="Face & Appearance",
                value=DEFAULT_FACE,
                lines=3,
                info="얼굴과 외모 묘사. 피부, 눈, 표정, 화장 등. 예: 'fair complexion, bright blue contact lenses, innocent expression'",
            )
            p_hair = gr.Textbox(
                label="Hair",
                value=DEFAULT_HAIR,
                lines=3,
                info="헤어스타일과 질감. 길이, 색상, 웨이브, 움직임 등. 예: 'long wavy jet-black hair flowing in the wind'",
            )
            p_outfit = gr.Textbox(
                label="Outfit",
                value=DEFAULT_OUTFIT,
                lines=3,
                info="의상과 소품 묘사. 옷 종류, 색상, 소재, 디테일 등. 예: 'tiny black lingerie set, delicate fabric'",
            )
            p_pose = gr.Textbox(
                label="Pose & Action",
                value=DEFAULT_POSE,
                lines=3,
                info="자세와 동작. 전신/반신, 시선 방향, 손발 위치 등. 예: 'full body, walking toward camera, arms hanging naturally'",
            )
            p_background = gr.Textbox(
                label="Background & Setting",
                value=DEFAULT_BACKGROUND,
                lines=3,
                info="배경 장소와 환경. 실내/실외, 계절, 주변 사물 등. 예: 'resort beach, white sand, ocean waves'",
            )
            p_lighting = gr.Textbox(
                label="Lighting",
                value=DEFAULT_LIGHTING,
                lines=2,
                info="조명 유형과 분위기. 자연광/인공광, 시간대, 색온도 등. 예: 'golden hour sunlight, soft warm highlights'",
            )
            p_camera = gr.Textbox(
                label="Camera & Shot Style",
                value=DEFAULT_CAMERA,
                lines=2,
                info="카메라 설정과 촬영 스타일. 렌즈, 심도, 앵글, 샷 유형 등. 예: '85mm portrait lens, shallow depth of field'",
            )
            p_quality = gr.Textbox(
                label="Quality & Technical",
                value=DEFAULT_QUALITY,
                lines=2,
                info="이미지 품질과 기술적 키워드. 해상도, 사실감, 렌더링 스타일 등. 예: 'ultra-realistic, 8k, photorealistic, sharp focus'",
            )
            p_anatomy = gr.Textbox(
                label="Anatomy",
                value=DEFAULT_ANATOMY,
                lines=2,
                info="해부학적 정확도 키워드. 손가락, 손, 발, 신체 비율 오류 방지 등. 예: 'correct finger count, no extra limbs'",
            )

            default_full_prompt = " ".join(
                [
                    DEFAULT_SUBJECT,
                    DEFAULT_FACE,
                    DEFAULT_HAIR,
                    DEFAULT_OUTFIT,
                    DEFAULT_POSE,
                    DEFAULT_BACKGROUND,
                    DEFAULT_LIGHTING,
                    DEFAULT_CAMERA,
                    DEFAULT_QUALITY,
                    DEFAULT_ANATOMY,
                ]
            )
            with gr.Accordion("Full Prompt", open=False):
                full_prompt = gr.Textbox(
                    label="", interactive=False, lines=8, value=default_full_prompt
                )

        # ── Right: Parameters + Output ────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### Generation Parameters")
            with gr.Row():
                width = gr.Slider(
                    256, 2048, value=768, step=64, label="Width",
                    info="Output image width in pixels (256–2048, step 64). FLUX.1-dev uses 64-pixel tile alignment.",
                )
                height = gr.Slider(
                    256, 2048, value=1536, step=64, label="Height",
                    info="Output image height in pixels (256–2048, step 64). FLUX.1-dev uses 64-pixel tile alignment.",
                )
            with gr.Row():
                guidance_scale = gr.Slider(
                    1.0, 20.0, value=4.0, step=0.5, label="Guidance Scale",
                    info="How closely the image follows the prompt. Higher = more prompt-adherent but less diverse. FLUX.1-dev works best at 4.0–15.0.",
                )
                num_inference_steps = gr.Slider(
                    10, 50, value=28, step=1, label="Steps",
                    info="Number of denoising steps. More steps = higher quality but slower. FLUX.1-dev works best at 20–28 steps.",
                )
            with gr.Row():
                seed = gr.Number(
                    value=42, label="Seed", precision=0,
                    info="Random seed for reproducibility. Use the same seed + settings to reproduce an image exactly.",
                )
                max_sequence_length = gr.Slider(
                    64, 512, value=512, step=64, label="Max Sequence Length",
                    info="T5-XXL text encoder max token length (64–512). Longer prompts need higher values. FLUX.1-dev supports up to 512.",
                )
            image_format = gr.Radio(
                ["JPEG", "PNG"], value="JPEG", label="Image Format",
                info="JPEG: quality 100, 4:4:4 subsampling (smaller file). PNG: lossless compression (larger file).",
            )
            btn = gr.Button("Generate", variant="primary")

            gr.Markdown("### Output")
            output_image = gr.Image(label="Output Image", type="pil", height=800)
            output_info = gr.Textbox(label="Info", interactive=False)

    btn.click(
        fn=generate,
        inputs=[
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
            height,
            width,
            guidance_scale,
            num_inference_steps,
            seed,
            max_sequence_length,
            image_format,
        ],
        outputs=[output_image, output_info],
    )

    def update_full_prompt(*sections):
        return " ".join(s for s in sections if s)

    section_inputs = [
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
    for section in section_inputs:
        section.change(
            fn=update_full_prompt, inputs=section_inputs, outputs=full_prompt
        )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
