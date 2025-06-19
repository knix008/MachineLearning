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
    def __init__(self, root_dir, domain="trainB", transform=None):
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
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
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
            nn.Conv2d(features[-1], features[-1] * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, 3, padding=1),
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
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # ↓ float32로 변환
        sqrt_alpha_bar = (
            torch.sqrt(torch.from_numpy(self.alpha_bar).to(x_start.device)[t])
            .view(-1, 1, 1, 1)
            .float()
        )
        sqrt_one_minus_alpha_bar = (
            torch.sqrt(1 - torch.from_numpy(self.alpha_bar).to(x_start.device)[t])
            .view(-1, 1, 1, 1)
            .float()
        )
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,))


def train(model, dataloader, scheduler, epochs=10, device="cuda"):
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    mse = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device).float()  # ← float32 변환 추가
            t = scheduler.sample_timesteps(imgs.size(0)).to(device)
            noise = torch.randn_like(imgs)
            x_noisy = scheduler.q_sample(imgs, t, noise)
            pred_noise = model(x_noisy, t)
            loss = mse(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Step {i}, Loss: {loss.item():.4f}")


def sample(model, scheduler, img_shape=(3, 128, 128), device="cuda"):
    model.eval()
    with torch.no_grad():
        img = torch.randn(1, *img_shape).to(device).float()  # ← float32 변환 추가
        for t in reversed(range(scheduler.timesteps)):
            t_batch = torch.tensor([t], device=device)
            pred_noise = model(img, t_batch)
            alpha = torch.tensor(scheduler.alpha[t], device=device).float()
            alpha_bar = torch.tensor(scheduler.alpha_bar[t], device=device).float()
            beta = torch.tensor(scheduler.beta[t], device=device).float()
            img = (1 / torch.sqrt(alpha)) * (
                img - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
            )
            if t > 0:
                noise = torch.randn_like(img)
                img = img + torch.sqrt(beta) * noise
        img = torch.clamp(img, -1, 1)
        img = (img + 1) / 2
        return img


# 5. 실행 예제
if __name__ == "__main__":
    data_root = "./data/selfie2anime"  # selfie2anime 데이터셋 경로
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_dataset = Selfie2AnimeDataset(data_root, domain="trainB", transform=transform)
    print(f"Number of training images: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    # UNet2d의 입력 채널: RGB 3채널 + timestep 1채널 = 4
    model = UNet2d(in_channels=4, out_channels=3).to(device)
    scheduler = DiffusionScheduler(timesteps=1000)

    train(model, train_loader, scheduler, epochs=10, device=device)

    sample_img = sample(model, scheduler, img_shape=(3, 128, 128), device=device)
    np_img = sample_img.squeeze().cpu().permute(1, 2, 0).numpy()
    plt.imshow(np_img)
    plt.title("Generated Anime Face")
    plt.axis("off")
    plt.show()
