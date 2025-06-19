import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from tqdm import tqdm
from PIL import Image

# ===== 1. 모델 정의 (Simple UNet 예시) =====
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1), nn.ReLU(inplace=True),    # 128 -> 64
            nn.Conv2d(features, features*2, 4, 2, 1), nn.BatchNorm2d(features*2), nn.ReLU(inplace=True), # 64 -> 32
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.BatchNorm2d(features*4), nn.ReLU(inplace=True), # 32 -> 16
            nn.Conv2d(features*4, features*8, 4, 2, 1), nn.BatchNorm2d(features*8), nn.ReLU(inplace=True) # 16 -> 8
        )
        # Middle block
        self.middle = nn.Sequential(
            nn.Conv2d(features*8, features*8, 3, 1, 1), nn.ReLU(inplace=True)
        )
        # Decoder: 8 -> 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1), nn.BatchNorm2d(features*4), nn.ReLU(inplace=True), # 8 -> 16
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1), nn.BatchNorm2d(features*2), nn.ReLU(inplace=True), # 16 -> 32
            nn.ConvTranspose2d(features*2, features, 4, 2, 1), nn.BatchNorm2d(features), nn.ReLU(inplace=True), # 32 -> 64
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1), # 64 -> 128
            nn.Tanh()
        )

    def forward(self, x, t=None):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        out = self.decoder(x2)
        return out

# ===== 2. Diffusion Utilities =====
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.device = device

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = self.alpha_hat[t][:, None, None, None].sqrt()
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t])[:, None, None, None].sqrt()
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def sample_timesteps(self, n):
        return torch.randint(0, self.timesteps, (n,), device=self.device).long()

# ===== 3. Custom Paired Dataset for trainA & trainB =====
class Selfie2AnimePairDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.trainA_dir = os.path.join(root_dir, "trainA")
        self.trainB_dir = os.path.join(root_dir, "trainB")
        self.A_images = sorted([
            os.path.join(self.trainA_dir, f) for f in os.listdir(self.trainA_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.B_images = sorted([
            os.path.join(self.trainB_dir, f) for f in os.listdir(self.trainB_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return min(len(self.A_images), len(self.B_images))

    def __getitem__(self, idx):
        imgA = Image.open(self.A_images[idx]).convert("RGB")
        imgB = Image.open(self.B_images[idx]).convert("RGB")
        imgA = self.transform(imgA)
        imgB = self.transform(imgB)
        return imgA, imgB

# ===== 4. 학습 루프 =====
def train_selfie2anime(
    train_dir="dataset/train",
    save_ckpt="ddpm_anime.pth",
    epochs=10,
    img_size=128,
    batch_size=16,
    lr=2e-4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    paired_dataset = Selfie2AnimePairDataset(train_dir, img_size=img_size)
    dataloader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    diffusion = Diffusion(1000, device=device)
    mse = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (imgA, imgB) in pbar:
            imgA = imgA.to(device) # 입력(사람얼굴)
            imgB = imgB.to(device) # 타겟(애니얼굴)
            t = diffusion.sample_timesteps(imgA.size(0))
            noise = torch.randn_like(imgB)
            x_t = diffusion.add_noise(imgB, t, noise=noise)
            predicted_noise = model(imgA, t)  # 사람얼굴을 입력받아 애니이미지 노이즈 예측
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
        # 에폭마다 모델 저장
        torch.save(model.state_dict(), save_ckpt)
    print(f"모델 저장 위치: {save_ckpt}")

if __name__ == "__main__":
    # 데이터셋은 dataset/train/trainA, dataset/train/trainB 폴더에 이미지가 있어야 합니다.
    train_selfie2anime(
        train_dir="data/selfie2anime",  # dataset 구조: dataset/train/trainA, dataset/train/trainB
        save_ckpt="data/ddpm_anime.pth",
        epochs=10,
        img_size=128,
        batch_size=16,
        lr=2e-4
    )