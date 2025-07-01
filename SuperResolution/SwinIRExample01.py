import gradio as gr
import torch
import numpy as np
from PIL import Image
import cv2

from SwinIR.models.network_swinir import SwinIR as net

# Downloaded from https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
MODEL_PATH = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"


def load_model():
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
        upsampler="nearest+conv",  # 반드시 classicalSR 체크포인트에는 이 값!
        resi_connection="1conv",
    )
    pretrained = torch.load(MODEL_PATH, weights_only=True)
    param_key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(
        pretrained[param_key] if param_key in pretrained else pretrained, strict=True
    )
    model.eval()
    return model


model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def preprocess(img_pil):
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img


def postprocess(output):
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-BGR -> HWC-RGB
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)


def swinir_upscale(img_pil):
    img_lq = preprocess(img_pil)
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
    return postprocess(output)


iface = gr.Interface(
    swinir_upscale,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=gr.Image(type="pil", label="Upscaled Image (SwinIR x4)"),
    title="SwinIR Image Upscaling (x4)",
    description="Upload an image to upscale by 4x using SwinIR (official model).",
)

if __name__ == "__main__":
    iface.launch()
