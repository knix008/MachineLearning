import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2
import time
import warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Run "git clone https://github.com/JingyunLiang/SwinIR"
# Run "pip install timm"
# Run "python download_swinir_weights.py" to download model weights

from SwinIR.models.network_swinir import SwinIR as net

TILE_SIZE = 512       # 타일 크기 (픽셀). 큰 이미지 처리 시 메모리 절약
TILE_OVERLAP = 32     # 타일 간 겹침 (경계 아티팩트 방지)
USE_FP16 = False      # FP16 autocast (SwinIR에서 black image 발생 가능, 기본 비활성화)
USE_COMPILE = False   # torch.compile 사용 (PyTorch 2.0+, 첫 실행 느림)

# 결과 저장 디렉토리 (스크립트와 동일한 디렉토리)
OUTPUT_DIR = Path(__file__).parent
DEFAULT_INPUT_IMAGE = "Test13.jpg"  # 기본 입력 이미지 경로 (적절히 변경)

# SwinIR 모델 프리셋 정의
MODEL_PRESETS = {
    "Real SR x4 GAN (SwinIR-M)": {
        "path": "weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "upscale": 4, "img_size": 64, "window_size": 8,
        "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": "nearest+conv", "resi_connection": "1conv",
    },
    "Real SR x4 PSNR (SwinIR-M)": {
        "path": "weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth",
        "upscale": 4, "img_size": 64, "window_size": 8,
        "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": "nearest+conv", "resi_connection": "1conv",
    },
    "Classical SR x4 (SwinIR-M)": {
        "path": "weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        "upscale": 4, "img_size": 64, "window_size": 8,
        "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": "pixelshuffle", "resi_connection": "1conv",
    },
    "Classical SR x2 (SwinIR-M)": {
        "path": "weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
        "upscale": 2, "img_size": 64, "window_size": 8,
        "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": "pixelshuffle", "resi_connection": "1conv",
    },
    "Custom": {
        "path": "", "upscale": 4, "img_size": 64, "window_size": 8,
        "depths": [6, 6, 6, 6, 6, 6], "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6], "mlp_ratio": 2,
        "upsampler": "nearest+conv", "resi_connection": "1conv",
    },
}

DEFAULT_PRESET = "Real SR x4 GAN (SwinIR-M)"


def load_model(device, cfg):
    model = net(
        upscale=cfg["upscale"],
        in_chans=3,
        img_size=cfg["img_size"],
        window_size=cfg["window_size"],
        img_range=1.0,
        depths=cfg["depths"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        upsampler=cfg["upsampler"],
        resi_connection=cfg["resi_connection"],
    )
    pretrained = torch.load(cfg["path"], weights_only=True)
    param_key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(
        pretrained[param_key] if param_key in pretrained else pretrained, strict=True
    )
    model.eval()
    return model.to(device)


# 디바이스 자동 감지
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
current_cfg = MODEL_PRESETS[DEFAULT_PRESET].copy()
model = load_model(device, current_cfg)

# FP16: 모델 가중치는 fp32 유지, autocast로 안전한 연산만 fp16 적용
if USE_FP16 and device.type == "cuda":
    print("FP16 autocast 모드 활성화 (모델 가중치는 fp32 유지)")

# torch.compile (PyTorch 2.0+, 첫 실행 시 컴파일 시간 소요)
if USE_COMPILE and hasattr(torch, "compile"):
    model = torch.compile(model)
    print("torch.compile 활성화")


def reload_model_ui(preset_name, custom_path, upscale, img_size, window_size,
                    embed_dim, depths_str, num_heads_str, mlp_ratio,
                    upsampler, resi_connection, tile_size, tile_overlap, use_fp16):
    """UI에서 모델 파라미터를 받아 모델을 재로드"""
    global model, current_cfg, TILE_SIZE, TILE_OVERLAP, USE_FP16

    try:
        if preset_name == "Custom":
            cfg = {
                "path": custom_path.strip(),
                "upscale": int(upscale),
                "img_size": int(img_size),
                "window_size": int(window_size),
                "embed_dim": int(embed_dim),
                "depths": [int(x.strip()) for x in depths_str.split(",")],
                "num_heads": [int(x.strip()) for x in num_heads_str.split(",")],
                "mlp_ratio": int(mlp_ratio),
                "upsampler": upsampler,
                "resi_connection": resi_connection,
            }
        else:
            cfg = MODEL_PRESETS[preset_name].copy()

        if not cfg["path"]:
            return "오류: 모델 경로가 비어 있습니다."

        print(f"모델 재로드 중: {cfg['path']}")
        model = load_model(device, cfg)
        current_cfg = cfg

        TILE_SIZE = int(tile_size)
        TILE_OVERLAP = int(tile_overlap)
        USE_FP16 = bool(use_fp16)

        if USE_FP16 and device.type == "cuda":
            print("FP16 autocast 모드 활성화")

        info = (f"모델 로드 완료!\n"
                f"  경로: {cfg['path']}\n"
                f"  Upscale: x{cfg['upscale']}  |  img_size: {cfg['img_size']}  |  window_size: {cfg['window_size']}\n"
                f"  embed_dim: {cfg['embed_dim']}  |  mlp_ratio: {cfg['mlp_ratio']}\n"
                f"  depths: {cfg['depths']}\n"
                f"  num_heads: {cfg['num_heads']}\n"
                f"  upsampler: {cfg['upsampler']}  |  resi_connection: {cfg['resi_connection']}\n"
                f"  Tile: {TILE_SIZE}px  |  Overlap: {TILE_OVERLAP}px  |  FP16: {USE_FP16}")
        print(info)
        return info
    except Exception as e:
        return f"오류: {e}"


def on_preset_change(preset_name):
    """프리셋 선택 시 파라미터 자동 채우기"""
    cfg = MODEL_PRESETS[preset_name]
    is_custom = preset_name == "Custom"
    return (
        gr.update(value=cfg["path"], interactive=is_custom),
        cfg["upscale"],
        cfg["img_size"],
        cfg["window_size"],
        cfg["embed_dim"],
        ", ".join(map(str, cfg["depths"])),
        ", ".join(map(str, cfg["num_heads"])),
        cfg["mlp_ratio"],
        cfg["upsampler"],
        cfg["resi_connection"],
    )


def preprocess(img_pil):
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    t = torch.from_numpy(img).unsqueeze(0).to(device)
    return t


def tile_upscale(img_lq, scale=4, window_size=8, progress_callback=None):
    """큰 이미지를 타일로 나눠 처리 (메모리 절약 + 대형 이미지 지원)"""
    b, c, h, w = img_lq.shape
    stride = TILE_SIZE - TILE_OVERLAP

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))
    total_tiles = len(y_steps) * len(x_steps)

    h_out, w_out = h * scale, w * scale
    output = torch.zeros(b, c, h_out, w_out, dtype=torch.float32, device=device)
    count = torch.zeros(b, 1, h_out, w_out, dtype=torch.float32, device=device)

    tile_idx = 0
    with tqdm(total=total_tiles, desc="타일 업스케일", unit="tile") as pbar:
        for y in y_steps:
            for x in x_steps:
                y_end = min(y + TILE_SIZE, h)
                x_end = min(x + TILE_SIZE, w)
                y_start = y_end - TILE_SIZE if y_end - y < TILE_SIZE else y
                x_start = x_end - TILE_SIZE if x_end - x < TILE_SIZE else x

                patch = img_lq[:, :, y_start:y_end, x_start:x_end]
                _, _, ph, pw = patch.shape
                h_pad = (ph // window_size + 1) * window_size - ph if ph % window_size != 0 else 0
                w_pad = (pw // window_size + 1) * window_size - pw if pw % window_size != 0 else 0
                if h_pad > 0 or w_pad > 0:
                    patch = torch.cat([patch, torch.flip(patch, [2])], 2)[:, :, :ph + h_pad, :]
                    patch = torch.cat([patch, torch.flip(patch, [3])], 3)[:, :, :, :pw + w_pad]

                with torch.autocast(device_type=device.type, enabled=(USE_FP16 and device.type == "cuda")):
                    out_patch = model(patch)
                out_patch = out_patch[..., :ph * scale, :pw * scale].float()

                oy, ox = y_start * scale, x_start * scale
                output[:, :, oy:oy + ph * scale, ox:ox + pw * scale] += out_patch
                count[:, :, oy:oy + ph * scale, ox:ox + pw * scale] += 1

                tile_idx += 1
                pbar.update(1)
                if progress_callback:
                    progress_callback(0.2 + 0.65 * tile_idx / total_tiles,
                                      desc=f"타일 업스케일 중... ({tile_idx}/{total_tiles})")

    output /= count
    return output


def postprocess(output):
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)


def swinir_upscale(img_pil, output_format, gr_progress=gr.Progress()):
    if img_pil is None:
        gr.Warning("이미지를 먼저 업로드해 주세요.")
        return None, "", ""

    gr_progress(0.0, desc="전처리 중...")
    print("전처리 중...")
    start_time = time.time()

    img_lq = preprocess(img_pil)

    gr_progress(0.2, desc="업스케일 중...")
    _, _, h_old, w_old = img_lq.size()
    scale = current_cfg["upscale"]
    window_size = current_cfg["window_size"]

    with torch.no_grad():
        if h_old <= TILE_SIZE and w_old <= TILE_SIZE:
            # 작은 이미지: 한 번에 처리
            h_pad = (h_old // window_size + 1) * window_size - h_old if h_old % window_size != 0 else 0
            w_pad = (w_old // window_size + 1) * window_size - w_old if w_old % window_size != 0 else 0
            img_padded = img_lq
            if h_pad > 0:
                img_padded = torch.cat([img_padded, torch.flip(img_padded, [2])], 2)[:, :, :h_old + h_pad, :]
            if w_pad > 0:
                img_padded = torch.cat([img_padded, torch.flip(img_padded, [3])], 3)[:, :, :, :w_old + w_pad]
            with tqdm(total=1, desc="업스케일 중", unit="image") as pbar:
                with torch.autocast(device_type=device.type, enabled=(USE_FP16 and device.type == "cuda")):
                    output = model(img_padded)
                pbar.update(1)
            output = output[..., :h_old * scale, :w_old * scale]
            gr_progress(0.85, desc="후처리 중...")
        else:
            # 큰 이미지: 타일 처리
            print(f"큰 이미지 ({w_old}x{h_old}) → 타일 처리 (tile={TILE_SIZE}, overlap={TILE_OVERLAP})")
            output = tile_upscale(img_lq, scale=scale, window_size=window_size,
                                  progress_callback=gr_progress)

    gr_progress(0.85, desc="후처리 중...")
    print("후처리 중...")
    result_img = postprocess(output)

    elapsed = time.time() - start_time
    elapsed_text = f"처리 시간: {elapsed:.2f}초"
    print(elapsed_text)

    # 파일 저장
    gr_progress(0.95, desc="파일 저장 중...")
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    script_name = Path(__file__).stem
    ext = "jpg" if output_format == "JPG" else "png"
    filename = f"{script_name}_{now}_{current_cfg['upscale']}x.{ext}"
    save_path = OUTPUT_DIR / filename
    if ext == "jpg":
        result_img.convert("RGB").save(save_path, format="JPEG", quality=95)
    else:
        result_img.save(save_path)
    print(f"저장 완료: {save_path}")

    gr_progress(1.0, desc="완료!")
    return result_img, elapsed_text, str(save_path)


_init_cfg = MODEL_PRESETS[DEFAULT_PRESET]

with gr.Blocks() as demo:
    gr.Markdown("# SwinIR Image Upscaling")
    gr.Markdown("Upload an image to upscale using SwinIR (official model).")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=500, value=DEFAULT_INPUT_IMAGE)
            with gr.Accordion("출력 설정", open=True):
                output_format = gr.Radio(
                    ["JPG", "PNG"],
                    value="JPG",
                    label="저장 파일 형식 ★JPG 권장",
                    info="JPG(권장): 용량 작음, quality 95 / PNG: 무손실"
                )
            run_btn = gr.Button("Upscale", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Upscaled Image", height=500)
            elapsed_text = gr.Textbox(label="처리 시간 (초)", interactive=False)
            save_path_text = gr.Textbox(label="저장 경로", interactive=False)

    with gr.Accordion("SwinIR 모델 파라미터 설정", open=False):
        gr.Markdown("설정 변경 후 **모델 재로드** 버튼을 눌러야 적용됩니다.")
        with gr.Row():
            preset_dd = gr.Dropdown(
                choices=list(MODEL_PRESETS.keys()),
                value=DEFAULT_PRESET,
                label="모델 프리셋",
                info="프리셋 선택 시 파라미터가 자동으로 채워집니다."
            )
        with gr.Group():
            gr.Markdown("### 모델 가중치 경로")
            custom_path = gr.Textbox(
                value=_init_cfg["path"],
                label="모델 파일 경로 (.pth)",
                interactive=(DEFAULT_PRESET == "Custom"),
                placeholder="weights/model.pth"
            )
        with gr.Group():
            gr.Markdown("### 아키텍처 파라미터 (가중치와 일치해야 함)")
            with gr.Row():
                upscale_num = gr.Number(value=_init_cfg["upscale"], label="upscale", precision=0, minimum=1, maximum=8)
                img_size_num = gr.Number(value=_init_cfg["img_size"], label="img_size", precision=0)
                window_size_num = gr.Number(value=_init_cfg["window_size"], label="window_size", precision=0)
                embed_dim_num = gr.Number(value=_init_cfg["embed_dim"], label="embed_dim", precision=0)
                mlp_ratio_num = gr.Number(value=_init_cfg["mlp_ratio"], label="mlp_ratio", precision=0)
            with gr.Row():
                depths_txt = gr.Textbox(value=", ".join(map(str, _init_cfg["depths"])), label="depths (쉼표 구분)")
                num_heads_txt = gr.Textbox(value=", ".join(map(str, _init_cfg["num_heads"])), label="num_heads (쉼표 구분)")
            with gr.Row():
                upsampler_dd = gr.Dropdown(
                    choices=["nearest+conv", "pixelshuffle", "pixelshuffledirect", ""],
                    value=_init_cfg["upsampler"],
                    label="upsampler"
                )
                resi_conn_dd = gr.Dropdown(
                    choices=["1conv", "3conv"],
                    value=_init_cfg["resi_connection"],
                    label="resi_connection"
                )
        with gr.Group():
            gr.Markdown("### 추론 파라미터")
            with gr.Row():
                tile_size_sl = gr.Slider(128, 1024, value=TILE_SIZE, step=64, label="Tile Size (px)")
                tile_overlap_sl = gr.Slider(0, 128, value=TILE_OVERLAP, step=8, label="Tile Overlap (px)")
                fp16_cb = gr.Checkbox(value=USE_FP16, label="FP16 (CUDA only, 블랙 이미지 주의)")
        with gr.Row():
            reload_btn = gr.Button("모델 재로드", variant="primary")
        reload_status = gr.Textbox(label="상태", interactive=False, lines=6)

    # 프리셋 변경 → 파라미터 자동 채우기
    preset_dd.change(
        on_preset_change,
        inputs=[preset_dd],
        outputs=[custom_path, upscale_num, img_size_num, window_size_num,
                 embed_dim_num, depths_txt, num_heads_txt, mlp_ratio_num,
                 upsampler_dd, resi_conn_dd],
    )

    # 모델 재로드
    reload_btn.click(
        reload_model_ui,
        inputs=[preset_dd, custom_path, upscale_num, img_size_num, window_size_num,
                embed_dim_num, depths_txt, num_heads_txt, mlp_ratio_num,
                upsampler_dd, resi_conn_dd, tile_size_sl, tile_overlap_sl, fp16_cb],
        outputs=[reload_status],
    )

    run_btn.click(swinir_upscale, inputs=[input_image, output_format], outputs=[output_image, elapsed_text, save_path_text])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
