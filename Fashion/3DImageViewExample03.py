import gradio as gr
import numpy as np
from PIL import Image
import torch
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import pyvista as pv
import cv2
import os

# 저장할 디렉토리 지정 (예: "html_outputs")
OUTPUT_DIR = "temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DPT 모델과 feature extractor 로드 (한 번만)
extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

def image_to_3d_surface_pv(input_img):
    # 입력 이미지를 DPT에 맞게 전처리
    inputs = extractor(images=input_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Depth map 후처리
    depth = predicted_depth.squeeze().cpu().numpy()
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth = (depth - depth_min) / (depth_max - depth_min)
    depth = cv2.resize(depth, (256, 256))
    depth *= 50  # 높이 스케일 조정

    # 3D surface mesh 생성
    x = np.arange(depth.shape[1])
    y = np.arange(depth.shape[0])
    x, y = np.meshgrid(x, y)
    z = depth

    grid = pv.StructuredGrid(x, y, z)

    # 고유 파일명 생성
    from datetime import datetime
    unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".html"
    html_path = os.path.join(OUTPUT_DIR, unique_name)

    # PyVista 3D plot을 HTML로 저장 (Panel 사용)
    plotter = pv.Plotter(window_size=(600, 600), off_screen=True)
    plotter.set_background("white")
    plotter.add_mesh(grid, cmap="plasma", show_edges=False)
    plotter.camera.elevation = 30
    plotter.export_html(html_path)
    plotter.close()

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content

# Gradio Custom HTML component
with gr.Blocks() as demo:
    gr.Markdown("## DPT 기반 2D → 3D 변환 (마우스로 3D 상호작용)\n이미지를 업로드하면 3D로 변환되어, 마우스로 상하좌우 돌려볼 수 있습니다.")
    with gr.Row():
        inp = gr.Image(type="pil", label="2D 이미지 업로드")
        out_html = gr.HTML(label="3D 모델(마우스 상호작용)")
    inp.change(image_to_3d_surface_pv, inputs=inp, outputs=out_html)

if __name__ == "__main__":
    demo.launch()