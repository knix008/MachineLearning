import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import math

# Restormer 모델 임포트 (Restormer 소스코드가 해당 경로에 있어야 합니다)
from restormer_arch import Restormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

restormer_params = {
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 48,
    'num_blocks': [4, 6, 6, 8],
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'dual_pixel_task': False
}

# 가중치 파일 경로를 환경에 맞게 지정하세요.
WEIGHT_PATH = "single_image_defocus_deblurring.pth"

def load_restormer(weights_path=WEIGHT_PATH):
    model = Restormer(**restormer_params)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model

model = load_restormer()

def pad_image(img, factor=8):
    w, h = img.size
    pad_w = (math.ceil(w / factor) * factor) - w
    pad_h = (math.ceil(h / factor) * factor) - h
    if pad_w == 0 and pad_h == 0:
        return img, (0, 0)
    img_np = np.array(img)
    img_pad = np.pad(
        img_np,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )
    img_pad = Image.fromarray(img_pad)
    return img_pad, (pad_w, pad_h)

def preprocess(img):
    img, (pw, ph) = pad_image(img)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return tensor, (pw, ph), img.size

def postprocess(tensor, orig_size):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor)
    img = img.crop((0, 0, orig_size[0], orig_size[1]))
    return img

def restormer_deblur(input_img):
    start = time.time()
    with torch.no_grad():
        inp, pad, orig_size = preprocess(input_img)
        restored = model(inp)
        if isinstance(restored, (list, tuple)):
            restored = restored[0]
        output_img = postprocess(restored, orig_size)
    elapsed = time.time() - start
    elapsed_str = f"처리 시간: {elapsed:.2f}초"
    return output_img, elapsed_str

demo = gr.Interface(
    fn=restormer_deblur,
    inputs=gr.Image(type="pil", label="흐린 사진 업로드"),
    outputs=[
        gr.Image(type="pil", label="Restormer 복원 결과"),
        gr.Textbox(label="이미지 처리 시간")
    ],
    title="Restormer: AI 이미지 디블러링 데모",
    description="Restormer 모델로 흐린 이미지를 자동 복원합니다. (Defocus Deblurring, CPU/GPU 지원, 8의 배수 자동 패딩/크롭)"
)

if __name__ == "__main__":
    demo.launch()