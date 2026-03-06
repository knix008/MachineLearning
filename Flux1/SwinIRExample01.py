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

from SwinIR.models.network_swinir import SwinIR as net

MODEL_PATH = "weights/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"

# 결과 저장 디렉토리 (스크립트와 동일한 디렉토리)
OUTPUT_DIR = Path(__file__).parent


def load_model(device):
    model = net(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )
    pretrained = torch.load(MODEL_PATH, weights_only=True)
    param_key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(
        pretrained[param_key] if param_key in pretrained else pretrained, strict=True
    )
    model.eval()
    return model.to(device)


# 디바이스 자동 감지
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
model = load_model(device)


def preprocess(img_pil):
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img


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
    with tqdm(total=1, desc="SwinIR Upscaling", unit="image") as pbar:
        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            window_size = 8
            scale = 4
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
                :, :, : h_old + h_pad, :
            ]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
                :, :, :, : w_old + w_pad
            ]
            output = model(img_lq)
            output = output[..., : h_old * scale, : w_old * scale]
        pbar.update(1)

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
    filename = f"{script_name}_{now}_4x.{ext}"
    save_path = OUTPUT_DIR / filename
    if ext == "jpg":
        result_img.convert("RGB").save(save_path, format="JPEG", quality=95)
    else:
        result_img.save(save_path)
    print(f"저장 완료: {save_path}")

    gr_progress(1.0, desc="완료!")
    return result_img, elapsed_text, str(save_path)


with gr.Blocks() as demo:
    gr.Markdown("# SwinIR Image Upscaling (x4)")
    gr.Markdown("Upload an image to upscale by 4x using SwinIR (official model).")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image", height=600)
            with gr.Accordion("Advanced Settings", open=True):
                output_format = gr.Radio(
                    ["JPG", "PNG"],
                    value="JPG",
                    label="저장 파일 형식 ★JPG 권장",
                    info="JPG(권장): 용량 작음, quality 95 / PNG: 무손실"
                )
            run_btn = gr.Button("Upscale", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Upscaled Image (SwinIR x4)", height=800)
            elapsed_text = gr.Textbox(label="처리 시간 (초)", interactive=False)
            save_path_text = gr.Textbox(label="저장 경로", interactive=False)
    run_btn.click(swinir_upscale, inputs=[input_image, output_format], outputs=[output_image, elapsed_text, save_path_text])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
