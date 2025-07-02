import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import gradio as gr
import time
import numpy as np

# ==================== 1. 데이터 로딩 ====================
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


# ==================== 2. 생성자와 판별자 ====================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.main(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.main(img_flat)
        return validity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 3. 훈련 함수 ====================
def train_gan(epochs=10, progress=gr.Progress()):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    start_time = time.time()
    fixed_noise = torch.randn(16, 100, device=device)
    epoch_images = []
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            real_imgs = imgs.to(device)

            # 1. 생성자 학습
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), 100, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # 2. 판별자 학습
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Gradio 진행률 업데이트
            if i % 100 == 0:
                progress(
                    (epoch + i / len(train_loader)) / epochs,
                    desc=f"Epoch {epoch+1}/{epochs}",
                )

        # Epoch별 생성 이미지 저장
        with torch.no_grad():
            gen_imgs = generator(fixed_noise).cpu()
        grid = np.transpose(
            torchvision.utils.make_grid(gen_imgs, nrow=4, normalize=True), (1, 2, 0)
        ).numpy()
        epoch_images.append(grid)

    elapsed_time = time.time() - start_time
    return epoch_images, f"{elapsed_time:.2f} seconds"


# ==================== 4. Gradio 인터페이스 ====================
def gradio_train(epochs):
    epoch_images, elapsed = train_gan(epochs)
    # Gradio Gallery에 맞는 포맷으로 변환
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
    title="Fashion MNIST 기반 패션 디자인 생성기",
    description="Fashion MNIST 데이터로 GAN을 학습하여 새로운 패션 디자인 이미지를 생성합니다. Epoch별로 생성 결과를 확인할 수 있습니다.",
    flagging_mode="never",
)

if __name__ == "__main__":
    iface.launch()
