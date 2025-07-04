import gradio as gr
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import plotly.graph_objs as go
from PIL import Image
import tempfile

# MiDaS 모델 로드 (최초 1회만)
def load_midas():
    model_type = "DPT_Large"  # or "MiDaS_small" for CPU
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type.startswith("DPT"):
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform, device

midas, midas_transform, device = load_midas()

def estimate_depth(image):
    # PIL.Image -> torch tensor
    input_batch = midas_transform(image).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    return depth

def depth_to_pointcloud(image, depth, sample_step=4):
    # 이미지를 numpy로 변환
    img = np.array(image)
    h, w = depth.shape
    # 샘플링 (속도/메모리 절약)
    xx, yy = np.meshgrid(np.arange(0, w, sample_step), np.arange(0, h, sample_step))
    zz = depth[yy, xx]
    colors = img[yy, xx] / 255.0
    return xx.flatten(), yy.flatten(), zz.flatten(), colors.reshape(-1, 3)

def plot_3d_pointcloud(x, y, z, colors):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2, color=colors, opacity=0.8),
        )
    ])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Depth',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Point Cloud from Single Image (MiDaS)"
    )
    # Plotly HTML을 임시파일로 저장 후 경로 반환
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
        fig.write_html(tmpfile.name)
        return tmpfile.name

def image_to_3d_view(image):
    depth = estimate_depth(image)
    x, y, z, colors = depth_to_pointcloud(image, depth)
    html_path = plot_3d_pointcloud(x, y, z, colors)
    return gr.HTML.update(value=open(html_path, encoding='utf8').read())

demo = gr.Interface(
    fn=image_to_3d_view,
    inputs=gr.Image(type="pil", label="2D 이미지 업로드"),
    outputs=gr.HTML(label="3D 포인트클라우드 시각화 (회전 가능)"),
    title="2D 이미지를 3D로 변환 (MiDaS)",
    description="이미지를 업로드하면 MiDaS로 깊이 추정 후 3D 포인트클라우드로 시각화합니다. 마우스로 회전/확대 가능합니다."
)

if __name__ == "__main__":
    demo.launch() 