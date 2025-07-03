import gradio as gr
import torch
from torchvision import transforms
import time
from collections import OrderedDict

# MPRNet 관련 코드
# https://github.com/swz30/MPRNet/blob/main/Models/MPRNet.py

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
    print("> 모델 가중치 로드 완료:", weights)
    return model


WEIGHT_PATH = "model_deblurring.pth"  # 모델 가중치 파일 경로
model = MPRNet()  # MPRNet 모델 인스턴스 생성
model = load_checkpoint(model, WEIGHT_PATH)  # 가중치 로드
model.eval()
model.to(device)


def pad_image(img):
    # 이미지 크기를 8의 배수로 패딩
    width, height = img.size
    new_width = (width + 7) // 8 * 8
    new_height = (height + 7) // 8 * 8
    img_pad = transforms.functional.pad(
        img, (0, 0, new_width - width, new_height - height), fill=0
    )
    return img_pad, (new_width, new_height)


def preprocess(img):
    img_pad, _ = pad_image(img)
    tensor = transforms.ToTensor()(img_pad).unsqueeze(0).to(device)
    return tensor, img.size  # img.size = (width, height)


def postprocess(tensor, orig_size):
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    img = transforms.ToPILImage()(tensor)
    # orig_size: (width, height)
    img = img.crop((0, 0, orig_size[0], orig_size[1]))
    return img


def deblur_ai(input_img):
    start_time = time.time()  # 시간 측정 시작
    with torch.no_grad():
        inp, orig_size = preprocess(input_img)
        restored = model(inp)
        if isinstance(restored, (list, tuple)):
            restored = restored[0]
        output_img = postprocess(restored, orig_size)
    end_time = time.time()  # 시간 측정 종료
    processing_time = end_time - start_time  # 처리 시간 계산
    return output_img, f"이미지 처리에 걸린 시간: {processing_time:.2f}초"


demo = gr.Interface(
    fn=deblur_ai,
    inputs=gr.Image(type="pil", label="흐린 사진 업로드"),
    outputs=[
        gr.Image(type="pil", label="AI가 선명하게 만든 사진"),
        gr.Textbox(label="이미지 처리에 걸린 시간"),
    ],
    title="AI 이미지 선명화 (디블러링)",
    description="딥러닝 기반 MPRNet 모델을 사용하여 흐린(블러) 사진을 자동으로 선명하게 바꿔줍니다. (CPU/GPU 모두 지원, 입력 이미지는 8의 배수로 자동 패딩됩니다.)",
)

if __name__ == "__main__":
    demo.launch()
