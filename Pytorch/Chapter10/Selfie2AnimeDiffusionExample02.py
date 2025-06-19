import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 정의 (Selfie2Anime)
class Selfie2AnimeDataset(Dataset):
    def __init__(self, root_dir, domain='trainB', transform=None):
        self.domain = domain
        self.dir = os.path.join(root_dir, domain)
        self.img_list = os.listdir(self.dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.img_list[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# 2. 표준 UNet2d 구현
class UNet2d(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder = nn.ModuleList()

        # 인코더
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = feature

        # 디코더
        for feature in reversed(features):
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1]*2, features[-1]*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upconvs = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        # t: timestep [batch]
        # timestep을 채널로 입력에 추가
        t = t[:, None, None, None].float()
        t = t.expand(x.shape[0], 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t], dim=1)

        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.decoder)):
            x = self.upconvs[idx](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)
        return self.final_conv(x)

# 3. DDPM Noise Scheduler
class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = np.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = torch.sqrt(torch.from_numpy(self.alpha_bar).to(x_start.device)[t]).view(-1, 1, 1, 1).float()
        sqrt_one_minus_alpha_bar = torch.sqrt(1-torch.from_numpy(self.alpha_bar).to(x_start.device)[t]).view(-1, 1, 1, 1).float()
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,))

# 4. 학습 함수
def train(model, dataloader, scheduler, epochs=20, device='cuda', save_path=None, print_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    mse = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device).float()
            t = scheduler.sample_timesteps(imgs.size(0)).to(device)
            noise = torch.randn_like(imgs)
            x_noisy = scheduler.q_sample(imgs, t, noise)
            pred_noise = model(x_noisy, t)
            loss = mse(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % print_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i}, Loss: {loss.item():.4f}")
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

# --- 노이즈에서 애니메이션 이미지 생성 + 중간과정 그리드 저장 ---
def sample_from_noise_with_interpolations(
    model, scheduler, img_shape=(3, 128, 128), device='cuda', save_grid_path="interpolation_grid.png",
    n_steps=8
):
    model.eval()
    with torch.no_grad():
        img = torch.randn(1, *img_shape).to(device).float()
        timesteps = np.linspace(scheduler.timesteps-1, 0, n_steps, dtype=int)
        images = []

        for idx, t in enumerate(reversed(range(scheduler.timesteps))):
            t_batch = torch.tensor([t], device=device)
            pred_noise = model(img, t_batch)
            alpha = torch.tensor(scheduler.alpha[t], device=device).float()
            alpha_bar = torch.tensor(scheduler.alpha_bar[t], device=device).float()
            beta = torch.tensor(scheduler.beta[t], device=device).float()
            img = (1 / torch.sqrt(alpha)) * (img - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise)
            if t > 0:
                noise = torch.randn_like(img)
                img = img + torch.sqrt(beta) * noise
            if t in timesteps:
                img_to_save = torch.clamp(img, -1, 1)
                img_to_save = (img_to_save + 1) / 2
                images.append(img_to_save.squeeze().cpu())

        # 만약 timesteps가 0을 포함하지 않으면 마지막 이미지를 추가
        if 0 not in timesteps:
            img_to_save = torch.clamp(img, -1, 1)
            img_to_save = (img_to_save + 1) / 2
            images.append(img_to_save.squeeze().cpu())

        # 그리드로 저장 및 시각화
        fig, axs = plt.subplots(1, len(images), figsize=(len(images)*3, 3))
        for i, img in enumerate(images):
            np_img = img.permute(1, 2, 0).numpy()
            axs[i].imshow(np.clip(np_img, 0, 1))
            axs[i].set_title(f"t={timesteps[i] if i < len(timesteps) else 0}")
            axs[i].axis('off')
        plt.suptitle("Progressive Anime Face Generation from Noise")
        plt.tight_layout()
        plt.savefig(save_grid_path)
        plt.show()

        # 개별 이미지 파일도 저장 (선택)
        for i, img in enumerate(images):
            np_img = img.permute(1, 2, 0).numpy()
            np_img = (np_img * 255).astype(np.uint8)
            img_pil = Image.fromarray(np_img)
            img_pil.save(f"interpolation_step_{timesteps[i] if i < len(timesteps) else 0}.png")

# 5. 실행 예제
def main():
    data_root = "./data/selfie2anime"  # selfie2anime 데이터셋 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = Selfie2AnimeDataset(data_root, domain='trainB', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    model = UNet2d(in_channels=4, out_channels=3).to(device)
    scheduler = DiffusionScheduler(timesteps=1000)
    
    # --- 모델 훈련 ---
    # epochs와 batch_size는 하드웨어 상황에 맞게 조정하세요.
    # 학습이 오래 걸릴 수 있습니다. 중간 저장 경로를 지정하면 epoch마다 저장됩니다.
    train(
        model, train_loader, scheduler,
        epochs=1,  # 실제로는 더 많은 epoch 권장
        device=device,
        save_path="unet2d_selfie2anime.pth"
    )

    # --- 노이즈에서 애니메이션 이미지 생성 및 중간과정 그리드 저장 ---
    sample_from_noise_with_interpolations(
        model, scheduler, img_shape=(3, 128, 128), device=device,
        save_grid_path="interpolation_grid.png", n_steps=8
    )
    
    
if __name__ == "__main__":
    main()  