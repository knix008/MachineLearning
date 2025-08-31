# generate_palm_vein.py
import torch
import legacy
import dnnlib
import numpy as np
from PIL import Image

network_pkl = './training-runs/00000-stylegan3-t-palm_vein/network-snapshot-XXXXXX.pkl'  # 학습된 모델 경로
device = torch.device('cuda')

with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
c = None  # 조건부 모델이 아니면 None

img = G(z, c, truncation_psi=0.7, noise_mode='const')
img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img = img[0].permute(1, 2, 0).cpu().numpy()
Image.fromarray(img, 'RGB').save('generated_palm_vein.png')