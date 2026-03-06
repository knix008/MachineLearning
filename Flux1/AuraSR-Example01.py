from aura_sr import AuraSR, tile_image, merge_tiles, create_checkerboard_weights, repeat_weights, create_offset_weights
from PIL import Image
import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import json
import math
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tqdm import tqdm
from datetime import datetime
from pathlib import Path as _Path
import os

# 결과 저장 디렉토리 (스크립트와 동일한 디렉토리)
OUTPUT_DIR = _Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 디바이스 자동 감지
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# 모델 로드 (device 직접 지정)
hf_model_path = Path(snapshot_download("fal/AuraSR-v2"))
config = json.loads((hf_model_path / "config.json").read_text())
aura_sr = AuraSR(config, device=device)
checkpoint = load_file(hf_model_path / "model.safetensors")
aura_sr.upsampler.load_state_dict(checkpoint, strict=True)

def run_upscale(aura_sr, tiles, max_batch_size, seed, gr_progress, progress_offset, progress_scale, label):
    """타일 배치 처리 + tqdm(CLI) + gr.Progress(Gradio) 진행률 표시"""
    device = aura_sr.upsampler.device
    batches = [tiles[i:i + max_batch_size] for i in range(0, len(tiles), max_batch_size)]
    reconstructed = []
    generator = torch.Generator(device=device).manual_seed(seed) if seed >= 0 else None
    for i, batch in enumerate(tqdm(batches, desc=label, unit="batch")):
        model_input = torch.stack(batch).to(device)
        with torch.no_grad():
            noise = torch.randn(model_input.shape[0], 128, device=device, generator=generator)
            output = aura_sr.upsampler(
                lowres_image=model_input,
                noise=noise,
            )
        reconstructed.extend(list(output.clamp_(0, 1).detach().cpu()))
        if gr_progress is not None:
            frac = (i + 1) / len(batches)
            gr_progress(progress_offset + frac * progress_scale, desc=f"{label} {i+1}/{len(batches)} 배치")
    return reconstructed


def upscale_image(input_image, scale, max_batch_size, weight_type, seed, output_format, gr_progress=gr.Progress()):
    if input_image is None:
        gr.Warning("이미지를 먼저 업로드해 주세요.")
        return None, ""
    image = input_image.convert("RGB")
    tensor_transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    input_size = aura_sr.input_image_size

    image_tensor = tensor_transform(image).unsqueeze(0)
    _, _, h, w = image_tensor.shape
    pad_h = (input_size - h % input_size) % input_size
    pad_w = (input_size - w % input_size) % input_size
    image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)

    wt = weight_type.replace(" ★권장", "")

    if scale == "4x (Overlapped) ★권장":
        gr_progress(0.0, desc="1패스 타일 분할 중...")
        tiles1, h1, w1 = tile_image(image_tensor, input_size)
        reconstructed1 = run_upscale(aura_sr, tiles1, max_batch_size, seed, gr_progress, 0.0, 0.45, "Pass 1")
        result1 = merge_tiles(reconstructed1, h1, w1, input_size * 4)

        gr_progress(0.45, desc="2패스 타일 분할 중...")
        offset = input_size // 2
        image_tensor_offset = F.pad(image_tensor, (offset, offset, offset, offset), mode='reflect').squeeze(0)
        tiles2, h2, w2 = tile_image(image_tensor_offset, input_size)
        reconstructed2 = run_upscale(aura_sr, tiles2, max_batch_size, seed, gr_progress, 0.45, 0.45, "Pass 2")
        result2 = merge_tiles(reconstructed2, h2, w2, input_size * 4)

        gr_progress(0.9, desc="타일 합성 중...")
        offset_4x = offset * 4
        result2_interior = result2[:, offset_4x:-offset_4x, offset_4x:-offset_4x]

        if wt == 'checkboard':
            weight_tile = create_checkerboard_weights(input_size * 4)
            weight_shape = result2_interior.shape[1:]
            weights_1 = create_offset_weights(weight_tile, weight_shape)
            weights_2 = repeat_weights(weight_tile, weight_shape)
            normalizer = weights_1 + weights_2
            weights_1 = (weights_1 / normalizer).unsqueeze(0).repeat(3, 1, 1)
            weights_2 = (weights_2 / normalizer).unsqueeze(0).repeat(3, 1, 1)
        else:
            weights_1 = torch.ones_like(result2_interior) * 0.5
            weights_2 = weights_1

        result = result1 * weights_2 + result2_interior * weights_1
        upscaled = to_pil(result[:, :h * 4, :w * 4])

    else:  # "4x"
        gr_progress(0.0, desc="타일 분할 중...")
        tiles, h_chunks, w_chunks = tile_image(image_tensor, input_size)
        reconstructed = run_upscale(aura_sr, tiles, max_batch_size, seed, gr_progress, 0.0, 0.9, "Upscaling")
        result = merge_tiles(reconstructed, h_chunks, w_chunks, input_size * 4)
        upscaled = to_pil(result[:, :h * 4, :w * 4])

    gr_progress(1.0, desc="완료!")

    # 파일명: 프로그램명_날짜_시간_스케일_배치크기_가중치타입.png
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    scale_tag = scale.replace(" ★권장", "").replace(" ", "-").replace("(", "").replace(")", "")
    wt_tag = wt
    batch_tag = f"batch{int(max_batch_size)}"
    seed_tag = f"seed{int(seed)}" if seed >= 0 else "seedR"
    ext = "jpg" if output_format == "JPG" else "png"
    filename = f"{_Path(__file__).stem}_{now}_{scale_tag}_{batch_tag}_{wt_tag}_{seed_tag}.{ext}"
    save_path = OUTPUT_DIR / filename
    if ext == "jpg":
        upscaled.convert("RGB").save(save_path, format="JPEG", quality=95)
    else:
        upscaled.save(save_path)
    print(f"저장 완료: {save_path}")

    return upscaled, str(save_path)

with gr.Blocks() as demo:
    gr.Markdown("# AuraSR Super Resolution Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=500)
            scale = gr.Radio(["4x (Overlapped) ★권장", "4x"], value="4x (Overlapped) ★권장", label="Upscale Factor")
            with gr.Accordion("Advanced Settings", open=True):
                max_batch_size = gr.Slider(minimum=1, maximum=32, value=8, step=1, label="Max Batch Size", info="권장: 8 | 한 번에 처리할 타일 수. 메모리 부족 시 줄이세요.")
                weight_type = gr.Radio(
                    ["checkboard ★권장", "constant"],
                    value="checkboard ★권장",
                    label="Weight Type (4x Overlapped 전용)",
                    info="checkboard(권장): 체크보드 가중치로 이음새 부드럽게 처리 / constant: 단순 0.5 평균"
                )
                seed = gr.Slider(minimum=-1, maximum=2147483647, value=-1, step=1, label="Noise Seed", info="-1: 매번 랜덤 (권장) | 0 이상: 고정 시드로 재현 가능한 결과")
                output_format = gr.Radio(
                    ["JPG", "PNG"],
                    value="JPG",
                    label="저장 파일 형식 ★JPG 권장",
                    info="JPG(권장): 용량 작음, quality 95 / PNG: 무손실"
                )
            run_btn = gr.Button("Upscale", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Upscaled Image")
            save_path_text = gr.Textbox(label="저장 경로", interactive=False)
    run_btn.click(upscale_image, inputs=[input_image, scale, max_batch_size, weight_type, seed, output_format], outputs=[output_image, save_path_text])

if __name__ == "__main__":
    demo.launch(inbrowser=True)