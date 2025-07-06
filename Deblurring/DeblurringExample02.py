import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr
import numpy as np
from collections import OrderedDict
from skimage import img_as_ubyte
from runpy import run_path
import os

MODEL_DIR = "./MPRNet/Deblurring"
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "pretrained_models", "model_deblurring.pth")
MODEL_CODE = os.path.join(MODEL_DIR, "MPRNet.py")

mprnet_module = run_path(MODEL_CODE)
MPRNet = mprnet_module["MPRNet"]


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MPRNet().to(device)
model = load_checkpoint(model, MODEL_WEIGHTS)
model.eval()


def mprnet_deblur_with_clamp_info(img_pil):
    img = img_pil.convert("RGB")
    input_ = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    input_ = input_.unsqueeze(0).to(device)

    img_multiple_of = 8
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

    with torch.no_grad():
        restored = model(input_)
    restored = restored[0]

    # 클램핑 전 min/max
    min_before = float(restored.min().cpu().numpy())
    max_before = float(restored.max().cpu().numpy())

    restored_clamped = torch.clamp(restored, 0, 1)

    # 클램핑 후 min/max
    min_after = float(restored_clamped.min().cpu().numpy())
    max_after = float(restored_clamped.max().cpu().numpy())

    restored_clamped = restored_clamped[:, :, :h, :w]
    restored_np = restored_clamped.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored_np = img_as_ubyte(restored_np[0])
    result_img = Image.fromarray(restored_np)

    clamp_report = (
        f"클램핑 전 min: {min_before:.6f}, max: {max_before:.6f}\n"
        f"클램핑 후 min: {min_after:.6f}, max: {max_after:.6f}"
    )
    return result_img, clamp_report


demo = gr.Interface(
    fn=mprnet_deblur_with_clamp_info,
    inputs=gr.Image(type="pil", label="Blurry Image"),
    outputs=[
        gr.Image(type="pil", label="Deblurred Image"),
        gr.Textbox(label="클램핑 min/max 정보"),
    ],
    title="MPRNet 디블러링 (클램핑 min/max 확인 포함)",
    description="이미지를 업로드하면 MPRNet으로 복원하고, 모델 출력의 클램핑 전후 min/max 값을 함께 보여줍니다.",
)

if __name__ == "__main__":
    demo.launch()
