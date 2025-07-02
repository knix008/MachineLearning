import gradio as gr
import torch
from torchvision import transforms, models
from PIL import Image
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_loader(image, size=256):
    loader = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def image_unloader(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().squeeze(0)
    return unloader(image)


def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class ContentLoss(torch.nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = torch.nn.functional.mse_loss(input, self.target)
        return input


class StyleLoss(torch.nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = torch.nn.functional.mse_loss(G, self.target)
        return input


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    content_layers=["conv_4"],
    style_layers=["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"],
):
    cnn = cnn.to(device).eval()
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = torch.nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, torch.nn.ReLU):
            name = f"relu_{i}"
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[: i + 1]
    return model, style_losses, content_losses


def style_transfer(
    content_img, style_img, num_steps=50, style_weight=1e6, content_weight=1
):
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    input_img = content_img.clone()
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img
    )
    input_img.requires_grad_(True)
    optimizer = torch.optim.LBFGS([input_img])
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            run[0] += 1
            return loss

        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img


def run_style_transfer(content, style, num_steps, style_weight, content_weight):
    content_img = image_loader(content)
    style_img = image_loader(style)
    start = time.time()
    output = style_transfer(
        content_img,
        style_img,
        num_steps=int(num_steps),
        style_weight=float(style_weight),
        content_weight=float(content_weight),
    )
    end = time.time()
    result_img = image_unloader(output)
    elapsed_time = f"처리 시간: {end - start:.2f}초"
    return result_img, elapsed_time


iface = gr.Interface(
    fn=run_style_transfer,
    inputs=[
        gr.Image(type="pil", label="콘텐츠 이미지"),
        gr.Image(type="pil", label="스타일 이미지"),
        gr.Slider(10, 1000, value=500, step=1, label="스텝 수 (num_steps)"),
        gr.Number(value=1e6, label="스타일 가중치 (style_weight)"),
        gr.Number(value=1, label="콘텐츠 가중치 (content_weight)"),
    ],
    outputs=[
        gr.Image(type="pil", label="스타일 트랜스퍼 결과"),
        gr.Textbox(label="처리 시간"),
    ],
    title="스타일 트랜스퍼 (Style Transfer)",
    description="콘텐츠 이미지와 스타일 이미지를 업로드하고, 스텝 수 및 가중치를 조절하세요. 스타일 트랜스퍼 결과와 처리 시간을 보여줍니다.",
)

if __name__ == "__main__":
    iface.launch()
