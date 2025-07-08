import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import trimesh
import tempfile
import os


def load_and_display_3d_model(file_path):
    """3D 모델 파일을 로드하고 Plotly로 시각화"""
    if file_path is None:
        return None

    try:
        # trimesh를 사용하여 3D 모델 로드
        mesh = trimesh.load(file_path)

        # 메시가 Scene 객체인 경우 첫 번째 geometry 추출
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]

        # 정점과 면 정보 추출
        vertices = mesh.vertices
        faces = mesh.faces

        # Plotly 3D 메시 생성
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color="lightblue",
                    opacity=0.8,
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.8,
                        roughness=0.2,
                        specular=0.6,
                        fresnel=0.2,
                    ),
                    lightposition=dict(x=100, y=200, z=300),
                )
            ]
        )

        # 레이아웃 설정
        fig.update_layout(
            title="3D Model Viewer",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=800,
            height=600,
        )
        return fig

    except Exception as e:
        return f"Error loading 3D model: {str(e)}"


def create_sample_3d_object():
    """샘플 3D 객체 생성 (구)"""
    # 구 생성
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure(
        data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis", showscale=False)]
    )

    fig.update_layout(
        title="Sample 3D Sphere",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=800,
        height=600,
    )

    return fig


def create_advanced_3d_viewer():
    """고급 3D 뷰어 생성"""
    with gr.Blocks(title="3D Model Viewer") as demo:
        gr.Markdown("# 3D Model Viewer")
        gr.Markdown(
            "Upload a 3D model file (.obj, .stl, .ply) and view it in 360 degrees!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload 3D Model",
                    file_types=[".obj", ".stl", ".ply", ".off"],
                    type="filepath",
                )

                sample_btn = gr.Button("Show Sample Sphere", variant="secondary")

                gr.Markdown("### Instructions:")
                gr.Markdown(
                    """
                - Upload a 3D model file (.obj, .stl, .ply)
                - Use mouse to rotate the model
                - Scroll to zoom in/out
                - Drag to pan
                """
                )

            with gr.Column(scale=2):
                output_plot = gr.Plot(label="3D Model")

        # 이벤트 핸들러
        file_input.change(
            fn=load_and_display_3d_model, inputs=file_input, outputs=output_plot
        )

        sample_btn.click(fn=create_sample_3d_object, inputs=None, outputs=output_plot)

    return demo


if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    print("Required packages:")
    print("pip install gradio plotly trimesh")

    # 데모 실행
    demo = create_advanced_3d_viewer()
    demo.launch(share=True)
