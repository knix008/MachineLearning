import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import plotly.graph_objects as go
from transformers import pipeline

# MiDaS 모델 로드 (Hugging Face Transformers 라이브러리 사용)
print("MiDaS 모델 로드 중...")
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
print("MiDaS 모델 로드 완료.")

def image_to_3d_point_cloud(image: Image.Image):
    """
    2D 이미지를 MiDaS를 사용하여 깊이 맵으로 변환하고,
    깊이 맵을 기반으로 3D 포인트 클라우드를 생성합니다.
    """
    # 깊이 추정
    # MiDaS pipeline은 딕셔너리를 반환하며, 'depth' 키에 깊이 맵이 있습니다.
    depth_prediction = depth_estimator(image)['depth']
    depth_map_np = np.array(depth_prediction)

    # 깊이 맵 정규화 (시각화를 위해 0-255 범위로)
    depth_map_display = cv2.normalize(depth_map_np, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_map_display = cv2.cvtColor(depth_map_display, cv2.COLOR_GRAY2BGR) # Gradio 출력을 위해 BGR로 변환

    # 이미지와 깊이 맵을 포인트 클라우드로 변환
    h, w = depth_map_np.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 픽셀 좌표를 (x, y, z)로 변환
    # z 값은 깊이 맵에서 가져오고, x, y는 이미지 좌표에 비례하게 설정
    # 깊이 맵 값이 작을수록 가까운 것으로 가정 (MiDaS 기본 동작)
    # x, y 스케일 조정으로 3D 모양 조절 가능
    x = (u - w / 2) / (w / 2)
    y = (v - h / 2) / (h / 2)
    z = depth_map_np
    
    # 색상 정보 추출 (원래 이미지에서)
    img_np = np.array(image)
    if img_np.shape[2] == 4: # RGBA 이미지인 경우 RGB로 변환
        img_np = img_np[:, :, :3]
    colors = img_np.reshape(-1, 3) / 255.0 # Plotly는 0-1 범위의 색상을 선호

    # 3D scatter plot 생성
    fig = go.Figure(data=[go.Scatter3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
            opacity=0.8
        )
    )])

    # 축 설정 (깊이 맵에 따라 z축 범위 조정)
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            aspectmode='data' # 축 비율 유지
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return depth_map_display, fig

# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=image_to_3d_point_cloud,
    inputs=gr.Image(type="pil", label="2D 이미지 업로드"),
    outputs=[
        gr.Image(type="numpy", label="추정된 깊이 맵"),
        gr.Plot(label="3D 포인트 클라우드 (드래그하여 회전)")
    ],
    title="2D 이미지를 3D 포인트 클라우드로 변환 (MiDaS + Plotly)",
    description="업로드된 2D 이미지에서 깊이 맵을 추정하고, 이를 기반으로 3D 포인트 클라우드를 생성하여 Plotly를 통해 360도 회전 가능한 뷰를 제공합니다."
)

iface.launch(share=True)