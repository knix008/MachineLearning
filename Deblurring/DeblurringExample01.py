import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr
import numpy as np
from collections import OrderedDict
from skimage import img_as_ubyte
import os
from runpy import run_path

# MPRNet 구성 및 가중치 경로를 환경에 맞게 지정하세요
MODEL_DIR = "./MPRNet/Deblurring"  # MPRNet.py가 있는 폴더
MODEL_WEIGHTS = os.path.join(MODEL_DIR, "pretrained_models", "model_deblurring.pth")
MODEL_CODE = os.path.join(MODEL_DIR, "MPRNet.py")

# MPRNet 모델 로드
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


def mprnet_deblur(img_pil):
    img = img_pil.convert("RGB")
    input_ = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    input_ = input_.unsqueeze(0).to(device)

    # Pad input so height/width are multiple of 8
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
    restored = torch.clamp(restored, 0, 1)

    # Unpad output
    restored = restored[:, :, :h, :w]

    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])
    result_img = Image.fromarray(restored)
    return result_img


demo = gr.Interface(
    fn=mprnet_deblur,
    inputs=gr.Image(type="pil", label="Blurry Image"),
    outputs=gr.Image(type="pil", label="Deblurred Image"),
    title="MPRNet 디블러링 데모",
    description="이미지를 업로드하면 MPRNet으로 선명하게 복원합니다. 모델 파일과 코드를 준비하세요.",
)

if __name__ == "__main__":
    demo.launch()
