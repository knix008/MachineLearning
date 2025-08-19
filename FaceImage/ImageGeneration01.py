import os
import sys
sys.path.append('./stylegan2-ada-pytorch')  # Update this path

import numpy as np
from PIL import Image
import torch
import dnnlib
import legacy

# Set output directory and number of images
output_dir = 'generated_faces'
num_images = 100_000
os.makedirs(output_dir, exist_ok=True)

# Path to pre-trained StyleGAN2-ADA model (download from NVIDIA's repo)
network_pkl = 'stylegan2-ffhq-config-f.pkl'  # Update with your model path

# Load pre-trained network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(network_pkl, 'rb') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

# Generate images
for i in range(num_images):
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    img = G(z, label, truncation_psi=0.7, noise_mode='const')
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img, 'RGB').save(os.path.join(output_dir, f'face_{i:05d}.png'))
    if (i+1) % 1000 == 0:
        print(f'Generated {i+1} images')

print('Done!')