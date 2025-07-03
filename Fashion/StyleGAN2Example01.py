import gradio as gr
import torch
import numpy as np
import time

# StyleGAN2 & e4e encoder 관련 임포트
from model import Generator                # StyleGAN2 모델
from utils.invert_image_with_e4e import invert_image  # e4e inversion 함수 (사용자 구현 필요)

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 512

# StyleGAN2 불러오기
g_ema = Generator(1024, latent_dim, 8).to(device)
g_ema.eval()
ckpt = torch.load("stylegan2-ffhq-config-f.pt", map_location=device)
g_ema.load_state_dict(ckpt["g_ema"], strict=False)

# e4e encoder 불러오기 (아래 경로는 예시, 실제 경로 확인 필요)
e4e_encoder = torch.load("e4e_ffhq_encode.pt", map_location=device)
e4e_encoder.eval()

def generate_similar_image(input_image, variation_strength=0.5):
    t0 = time.time()
    # 1. 이미지 latent vector로 inversion (e4e 사용)
    latent = invert_image(input_image, e4e_encoder, device=device)  # shape: (1, 512)
    # 2. latent 주변에서 노이즈 추가
    noise = torch.randn_like(latent) * variation_strength
    latent_new = latent + noise
    # 3. StyleGAN2로 이미지 생성
    with torch.no_grad():
        img, _ = g_ema([latent_new])
    img = (img.clamp(-1, 1) + 1) / 2
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    dt = time.time() - t0
    return img_np, f"이미지 생성에 걸린 시간: {dt:.2f}초"

demo = gr.Interface(
    fn=generate_similar_image,
    inputs=[
        gr.Image(type="pil", label="기준 이미지 업로드"),
        gr.Slider(0.0, 1.0, value=0.5, label="변형 강도 (0=매우 유사, 1=많이 다름)")
    ],
    outputs=[
        gr.Image(type="numpy", label="유사 이미지"),
        gr.Textbox(label="생성 시간"),
    ],
    title="StyleGAN2 유사 이미지 생성기",
    description="특정 이미지를 업로드하면, StyleGAN2 latent space에서 inversion 후 근처 latent에서 이미지를 생성해줍니다. (학습된 weight 필요)",
    examples=None
)

if __name__ == "__main__":
    demo.launch()