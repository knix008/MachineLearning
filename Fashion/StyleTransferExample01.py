import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import gradio as gr
import time # 시간 측정을 위한 time 모듈 임포트
import numpy as np # 예시 이미지 생성을 위한 numpy 임포트

# --- 신경망 스타일 전이 핵심 함수 (이전 코드에서 복사 및 수정) ---

# CUDA 또는 CPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, img_size=(512, 512)):
    """이미지를 불러와 PyTorch 텐서로 전처리합니다."""
    loader = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.406])
    ])
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

# 역정규화 변환 (텐서를 PIL Image로 변환할 때 사용)
unloader = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.ToPILImage()
])

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device)
        self.features = self.features[:36] # conv5_4까지 사용

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        self.name_map = {
            '0': 'conv1_1', '2': 'conv1_2',
            '5': 'conv2_1', '7': 'conv2_2',
            '10': 'conv3_1', '12': 'conv3_2', '14': 'conv3_3', '16': 'conv3_4',
            '19': 'conv4_1', '21': 'conv4_2', '23': 'conv4_3', '25': 'conv4_4',
            '28': 'conv5_1', '30': 'conv5_2', '32': 'conv5_3', '34': 'conv5_4',
        }
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = {}
        for name, layer in enumerate(self.features):
            x = layer(x)
            if str(name) in self.name_map:
                layer_name = self.name_map[str(name)]
                if layer_name in self.content_layers or layer_name in self.style_layers:
                    features[layer_name] = x
        return features

def gram_matrix(tensor):
    """그람 행렬을 계산하여 스타일 특징을 나타냅니다."""
    a, b, c, d = tensor.size()
    features = tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def run_style_transfer(content_img_path, style_img_path, num_steps, style_weight, content_weight, step_interval_to_save=50):
    """
    신경망 기반 스타일 전이를 실행하고 결과 이미지를 반환합니다.
    각 스텝별 중간 이미지를 저장하고, 처리 시간을 반환합니다.
    """
    if content_img_path is None or style_img_path is None:
        return None, "콘텐츠 이미지와 스타일 이미지를 모두 업로드해주세요.", None

    intermediate_images = [] # 중간 이미지들을 저장할 리스트

    try:
        content_image = load_image(content_img_path)
        style_image = load_image(style_img_path)
        generated_image = content_image.clone().requires_grad_(True)
        model = VGG().eval()
        optimizer = optim.LBFGS([generated_image])
        start_time = time.time() # 시작 시간 기록

        print(f"스타일 전이 시작 (Steps: {num_steps}, Style Weight: {style_weight}, Content Weight: {content_weight})")
        run = [0]
        while run[0] <= num_steps:
            def closure():
                optimizer.zero_grad()
                gen_features = model(generated_image)
                content_features = model(content_image)
                style_features = model(style_image)
                content_loss = 0
                style_loss = 0

                for layer in model.content_layers:
                    content_loss += torch.mean((gen_features[layer] - content_features[layer])**2)

                for layer in model.style_layers:
                    gen_gram = gram_matrix(gen_features[layer])
                    style_gram = gram_matrix(style_features[layer])
                    style_loss += torch.mean((gen_gram - style_gram)**2)

                total_loss = content_weight * content_loss + style_weight * style_loss
                total_loss.backward()

                # 각 스텝에서 생성된 이미지 저장
                if run[0] % step_interval_to_save == 0 or run[0] == 1:
                    current_gen_img = generated_image.clone().squeeze(0)
                    current_gen_img = torch.clamp(current_gen_img, 0, 1) # 0과 1 사이로 클리핑
                    pil_img = unloader(current_gen_img.cpu())
                    intermediate_images.append(pil_img)
                    print(f"Step {run[0]}: Total Loss: {total_loss.item():.4f}")

                run[0] += 1
                return total_loss

            optimizer.step(closure)

        end_time = time.time() # 종료 시간 기록
        processing_time = end_time - start_time # 처리 시간 계산

        # 최종 결과 이미지 (intermediate_images 리스트의 마지막 이미지)
        final_image = intermediate_images[-1] if intermediate_images else None
        status_message = f"스타일 전이가 완료되었습니다! 총 처리 시간: {processing_time:.2f} 초"
    
        # 중간 이미지 리스트와 상태 메시지, 처리 시간을 반환
        return final_image, status_message, intermediate_images

    except Exception as e:
        error_message = f"오류 발생: {e}. GPU 메모리가 부족하거나 이미지 파일에 문제가 있을 수 있습니다."
        print(error_message)
        return None, error_message, []

# --- 3. Gradio 인터페이스 구축 ---

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=run_style_transfer,
    inputs=[
        gr.Image(type="filepath", label="콘텐츠 이미지 (예: 티셔츠)", show_label=True),
        gr.Image(type="filepath", label="스타일 이미지 (예: 꽃무늬 패턴, 명화)", show_label=True),
        gr.Slider(minimum=50, maximum=500, step=50, value=200, label="최적화 단계 수 (Steps)", info="높을수록 결과가 좋지만, 시간이 오래 걸립니다."),
        gr.Slider(minimum=1e4, maximum=1e7, step=1e4, value=1e6, label="스타일 가중치 (Style Weight)", info="높을수록 스타일 이미지가 강하게 반영됩니다."),
        gr.Slider(minimum=1e0, maximum=1e3, step=1e0, value=1e1, label="콘텐츠 가중치 (Content Weight)", info="높을수록 콘텐츠 이미지의 원본 형태가 유지됩니다."),
        gr.Slider(minimum=10, maximum=100, step=10, value=50, label="중간 이미지 저장 간격 (Steps)", info="이 스텝 간격마다 중간 이미지를 저장합니다.")
    ],
    outputs=[
        gr.Image(type="pil", label="최종 패션 디자인", show_label=True),
        gr.Textbox(label="상태 메시지 및 처리 시간"),
        gr.Gallery(label="단계별 생성 이미지", preview=True, columns=4, rows=2, object_fit="contain", height="auto") # 갤러리 컴포넌트 추가
    ],
    title="신경망 패션 스타일 전이",
    description="콘텐츠 이미지(예: 옷)에 스타일 이미지(예: 패턴 또는 예술 작품)의 스타일을 적용하여 새로운 패션 디자인을 생성합니다. GPU가 있으면 훨씬 빠르게 작동합니다.",
    theme="soft"
)

# Gradio 앱 실행
if __name__ == "__main__":
    print(f"Gradio 앱이 '{device}'에서 실행됩니다.")
    iface.launch() # share=True를 추가하면 공개 링크 생성 가능 (일시적)