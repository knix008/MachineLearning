import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ===== SimpleUNet (128x128 호환) =====
class SimpleUNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, features, 4, 2, 1), torch.nn.ReLU(inplace=True),    # 128 -> 64
            torch.nn.Conv2d(features, features*2, 4, 2, 1), torch.nn.BatchNorm2d(features*2), torch.nn.ReLU(inplace=True), # 64 -> 32
            torch.nn.Conv2d(features*2, features*4, 4, 2, 1), torch.nn.BatchNorm2d(features*4), torch.nn.ReLU(inplace=True), # 32 -> 16
            torch.nn.Conv2d(features*4, features*8, 4, 2, 1), torch.nn.BatchNorm2d(features*8), torch.nn.ReLU(inplace=True) # 16 -> 8
        )
        self.middle = torch.nn.Sequential(
            torch.nn.Conv2d(features*8, features*8, 3, 1, 1), torch.nn.ReLU(inplace=True)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(features*8, features*4, 4, 2, 1), torch.nn.BatchNorm2d(features*4), torch.nn.ReLU(inplace=True), # 8 -> 16
            torch.nn.ConvTranspose2d(features*4, features*2, 4, 2, 1), torch.nn.BatchNorm2d(features*2), torch.nn.ReLU(inplace=True), # 16 -> 32
            torch.nn.ConvTranspose2d(features*2, features, 4, 2, 1), torch.nn.BatchNorm2d(features), torch.nn.ReLU(inplace=True), # 32 -> 64
            torch.nn.ConvTranspose2d(features, out_channels, 4, 2, 1), # 64 -> 128
            torch.nn.Tanh()
        )

    def forward(self, x, t=None):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        out = self.decoder(x2)
        return out

# ===== Diffusion Utilities =====
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.device = device

# ===== 얼굴 → 애니 변환 함수 =====
def face2anime(image_path, model_ckpt, output_path="anime_result.png", img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 네트워크는 입력 이미지를 바로 애니 이미지로 매핑하도록 훈련됨
    with torch.no_grad():
        anime_tensor = model(img_tensor)
    out_img = (anime_tensor.clamp(-1, 1) + 1) / 2
    out_img_pil = transforms.ToPILImage()(out_img.squeeze().cpu())

    out_img_pil.save(output_path)
    print(f"변환된 애니 이미지 저장: {output_path}")

    plt.subplot(1,2,1)
    plt.title("Input")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Anime Output")
    plt.imshow(out_img_pil)
    plt.axis('off')
    plt.show()

def main():
    # 예시로 사용할 이미지 경로와 모델 체크포인트 경로
    image_path = "sample_face.jpg"  # 변환할 얼굴 이미지 경로
    model_ckpt = "data/ddpm_anime.pth"  # 훈련된 모델 체크포인트 경로
    output_path = "anime_face.png"  # 결과 이미지 저장 경로

    face2anime(image_path, model_ckpt, output_path)

if __name__ == "__main__":
    print("Selfie to Anime 변환 예제...")
    main()

# ===== 사용 예시 =====
# face2anime("your_face.jpg", "data/ddpm_anime.pth", output_path="anime_face.png")