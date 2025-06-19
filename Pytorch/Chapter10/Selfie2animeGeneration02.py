import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# --- Dataset ---
class PairedDataset(Dataset):
    def __init__(self, real_dir, anime_dir, transform=None):
        self.real_files = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('jpg', 'png'))])
        self.anime_files = sorted([os.path.join(anime_dir, f) for f in os.listdir(anime_dir) if f.endswith(('jpg', 'png'))])
        self.transform = transform

    def __len__(self):
        return min(len(self.real_files), len(self.anime_files))

    def __getitem__(self, idx):
        real_img = Image.open(self.real_files[idx]).convert('RGB')
        anime_img = Image.open(self.anime_files[idx]).convert('RGB')
        if self.transform:
            real_img = self.transform(real_img)
            anime_img = self.transform(anime_img)
        return real_img, anime_img

# --- UNet Generator ---
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Downsampling
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)
        # Upsampling
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

# --- PatchGAN Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)

# --- Training ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 256
    batch_size = 4
    epochs = 20

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PairedDataset('data/selfie2anime/trainA', 'data/selfie2anime/trainB', transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = UNetGenerator().to(device)
    D = Discriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real, anime) in enumerate(tqdm(loader)):
            real = real.to(device)
            anime = anime.to(device)
            # Forward pass through discriminator to get output shape
            with torch.no_grad():
                pred_shape = D(real, anime).shape
            valid = torch.ones(pred_shape, device=device)
            fake = torch.zeros(pred_shape, device=device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            fake_anime = G(real)
            pred_fake = D(real, fake_anime)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_L1 = criterion_L1(fake_anime, anime)
            loss_G = loss_GAN + 100 * loss_L1
            loss_G.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            pred_real = D(real, anime)
            loss_real = criterion_GAN(pred_real, valid)
            pred_fake = D(real, fake_anime.detach())
            loss_fake = criterion_GAN(pred_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}] Loss_G: {loss_G.item():.4f} Loss_D: {loss_D.item():.4f}")

        # Save sample output
        utils.save_image((fake_anime * 0.5 + 0.5), f"sample_fake_epoch{epoch}.png")

    torch.save(G.state_dict(), "unet_generator_selfie2anime.pth")

if __name__ == "__main__":
    train()
