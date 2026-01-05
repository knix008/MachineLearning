import os
import sys
import torch
import numpy as np
from PIL import Image
import gradio as gr
import time
import warnings  # 추가

# 모든 UserWarning 무시
warnings.filterwarnings("ignore", category=UserWarning)

# $ git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
# $ curl -L https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -o stylegan2-ffhq-config-f.pkl

sys.path.append("./stylegan2-ada-pytorch")  # Update this path
import dnnlib
import legacy

def generate_images(network_pkl, output_dir, num_images, truncation_psi, noise_mode):
    # Automatically select device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        z_dtype = np.float64
        label_dtype = torch.float64
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        z_dtype = np.float32
        label_dtype = torch.float32
    else:
        device = torch.device("cpu")
        z_dtype = np.float32
        label_dtype = torch.float32
    os.makedirs(output_dir, exist_ok=True)
    # Load pre-trained network
    with open(network_pkl, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    image_paths = []
    start = time.time()
    for i in range(num_images):
        z = torch.from_numpy(np.random.randn(1, G.z_dim).astype(z_dtype)).to(device)
        label = torch.zeros([1, G.c_dim], device=device, dtype=label_dtype)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img_path = os.path.join(output_dir, f"10{i:05d}.jpg")
        Image.fromarray(img).save(img_path)
        image_paths.append(img_path)
        if (i + 1) % 100 == 0:
            print(f"Generated {i+1} images")

    end = time.time()
    elapsed = end - start
    print(f"Image generation completed in {elapsed:.2f} seconds")
    return image_paths[: min(4, len(image_paths))]  # 최대 4개 미리보기


iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(
            label="Network pkl 경로",
            value="stylegan2-ffhq-config-f.pkl",
            info="사용할 StyleGAN2 네트워크의 pkl 파일 경로를 입력합니다. 예: stylegan2-ffhq-config-f.pkl",
        ),
        gr.Textbox(
            label="출력 폴더",
            value="generated_faces",
            info="생성된 이미지가 저장될 폴더명을 입력합니다. 폴더가 없으면 자동 생성됩니다.",
        ),
        gr.Slider(
            1,
            100000,
            value=4,
            step=1,
            label="생성할 이미지 개수",
            info="한 번에 생성할 이미지의 개수를 지정합니다. (1~100000)",
        ),
        gr.Slider(
            0.0,
            1.0,
            value=0.7,
            step=0.01,
            label="truncation_psi",
            info="이미지 다양성과 품질을 조절하는 값입니다. 1.0에 가까울수록 다양성이 높고, 0에 가까울수록 품질이 높아집니다.",
        ),
        gr.Dropdown(
            ["const", "random", "none"],
            value="const",
            label="noise_mode",
            info="노이즈 적용 방식을 선택합니다. const: 고정된 노이즈, random: 매번 다른 노이즈, none: 노이즈 미적용",
        ),
    ],
    outputs=[gr.Gallery(label="생성된 이미지 (최대 4개 미리보기)")],
    title="StyleGAN2 얼굴 이미지 생성기",
    description="네트워크 pkl, 출력 폴더, 이미지 개수, truncation_psi, noise_mode를 설정해 얼굴 이미지를 생성합니다.",
)

if __name__ == "__main__":
    iface.launch(inbrowser=True)
