import gradio as gr
import torch
from torchvision import transforms
from collections import OrderedDict

# MPRNet 관련 코드
# https://github.com/swz30/MPRNet/blob/main/Models/MPRNet.py
# https://github.com/swz30/MPRNet/blob/main/utils.py

from MPRNet import MPRNet  # MPRNet 모델 클래스 import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


WEIGHT_PATH = "model_deblurring.pth"  # 모델 가중치 파일 경로
model = MPRNet()  # MPRNet 모델 인스턴스 생성
model = load_checkpoint(model, WEIGHT_PATH)  # 가중치 로드
model.eval()
model.to(device)


def preprocess(img):
    # PIL Image -> torch tensor [1, C, H, W], normalized to [0,1]
    img = img.convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def postprocess(tensor):
    # torch tensor [1, C, H, W] -> PIL Image, clipped to [0,255]
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor)
    return img


def deblur_ai(input_img):
    with torch.no_grad():
        inp = preprocess(input_img)
        restored = model(inp)
        if isinstance(restored, (list, tuple)):
            restored = restored[0]
        output_img = postprocess(restored)
    return output_img


demo = gr.Interface(
    fn=deblur_ai,
    inputs=gr.Image(type="pil", label="흐린 사진 업로드"),
    outputs=gr.Image(type="pil", label="AI가 선명하게 만든 사진"),
    title="AI 이미지 선명화 (디블러링)",
    description="딥러닝 기반 MPRNet 모델을 사용하여 흐린(블러) 사진을 자동으로 선명하게 바꿔줍니다. (CPU/GPU 모두 지원)",
)

if __name__ == "__main__":
    demo.launch()
