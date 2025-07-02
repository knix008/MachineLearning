import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import gradio as gr
import time
import numpy as np

# 1. 데이터 로딩
transform = transforms.Compose(
    [
        transforms.Resize(32),  # DCGAN은 32x32로 많이 사용
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# 2. DCGAN Generator & Discriminator
class DCGAN_G(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),  # output: 1 x 32 x 32
        )

    def forward(self, z):
        return self.main(z)


class DCGAN_D(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(ndf * 4 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3. 훈련 함수 (DCGAN)
def train_dcgan(epochs=10, progress=gr.Progress()):
    nz = 100
    netG = DCGAN_G(nz=nz).to(device)
    netD = DCGAN_D().to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)
    epoch_images = []
    start_time = time.time()

    for epoch in range(epochs):
        for i, (real, _) in enumerate(train_loader):
            real = real.to(device)
            b_size = real.size(0)
            label_real = torch.full((b_size,), 0.9, device=device)  # soft label
            label_fake = torch.zeros(b_size, device=device)

            # 1. 판별자
            netD.zero_grad()
            output = netD(real).view(-1)
            errD_real = criterion(output, label_real)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach()).view(-1)
            errD_fake = criterion(output_fake, label_fake)
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # 2. 생성자
            netG.zero_grad()
            label_gen = torch.full((b_size,), 1.0, device=device)
            output_gen = netD(fake).view(-1)
            errG = criterion(output_gen, label_gen)
            errG.backward()
            optimizerG.step()

            # Gradio 진행률
            if i % 100 == 0:
                progress(
                    (epoch + i / len(train_loader)) / epochs,
                    desc=f"Epoch {epoch+1}/{epochs}",
                )

        # Epoch별 생성 이미지 저장
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid = np.transpose(
            torchvision.utils.make_grid(fake, nrow=4, normalize=True), (1, 2, 0)
        ).numpy()
        epoch_images.append(grid)

    elapsed_time = time.time() - start_time
    return epoch_images, f"{elapsed_time:.2f} seconds"


# 4. Gradio 인터페이스
def gradio_train(epochs):
    epoch_images, elapsed = train_dcgan(epochs)
    gallery_imgs = [(img, f"{i+1} epoch") for i, img in enumerate(epoch_images)]
    return gallery_imgs, elapsed


iface = gr.Interface(
    fn=gradio_train,
    inputs=gr.Number(value=5, label="Epoch 수"),
    outputs=[
        gr.Gallery(
            label="Epoch별 생성된 패션 디자인",
            show_label=True,
            columns=4,
            rows=2,
            height="auto",
        ),
        gr.Text(label="소요 시간(초)"),
    ],
    title="Fashion MNIST DCGAN 기반 패션 디자인 생성기",
    description="Fashion MNIST 데이터로 DCGAN을 학습하여 더 나은 품질의 패션 디자인 이미지를 생성합니다. Epoch별로 생성 결과를 확인할 수 있습니다.",
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()
