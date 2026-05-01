import os
import subprocess
import sys
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


DEVICE = torch.device("cpu")
MODEL_REPO = "stabilityai/TripoSR"
TSR_REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
TSR_LOCAL_DIR = Path(__file__).resolve().parent / "third_party" / "TripoSR"
ORIENTATION_TRANSFORMS = {
    "자동 보정 (세우기)": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "원본 방향": None,
    "반대 방향 보정": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "180도 회전": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
}


def ensure_tsr_module() -> None:
    tsr_pkg_dir = TSR_LOCAL_DIR / "tsr"
    if not tsr_pkg_dir.exists():
        TSR_LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", TSR_REPO_URL, str(TSR_LOCAL_DIR)],
            check=True,
        )

    tsr_root = str(TSR_LOCAL_DIR)
    if tsr_root not in sys.path:
        sys.path.insert(0, tsr_root)


ensure_tsr_module()

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


@lru_cache(maxsize=1)
def get_model() -> TSR:
    model = TSR.from_pretrained(
        MODEL_REPO,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(str(DEVICE))
    return model


def get_device_text() -> str:
    return "CPU 고정 모드 사용 중"


def _prepare_image(
    image: Image.Image,
    do_remove_bg: bool,
    foreground_ratio: float,
) -> Image.Image:
    image = image.convert("RGB")
    if do_remove_bg:
        image = remove_background(image)
        image = resize_foreground(image, foreground_ratio)
        image = image.convert("RGB")
    return image


def _apply_orientation(mesh, orientation: str):
    oriented_mesh = mesh.copy()
    transform = ORIENTATION_TRANSFORMS.get(orientation)
    if transform is not None:
        oriented_mesh.apply_transform(transform)
    return oriented_mesh


def image_to_3d(
    image: Image.Image,
    remove_bg: bool,
    foreground_ratio: float,
    mesh_resolution: int,
    orientation: str,
) -> tuple[str, str, str]:
    if image is None:
        raise gr.Error("이미지를 먼저 업로드해 주세요.")

    model = get_model()
    image = _prepare_image(image, remove_bg, foreground_ratio)

    with torch.no_grad():
        scene_codes = model([image], device=str(DEVICE))
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=mesh_resolution,
        )

    mesh = _apply_orientation(meshes[0], orientation)
    tmp_dir = Path(tempfile.mkdtemp(prefix="triposr_"))
    obj_path = tmp_dir / "result.obj"
    glb_path = tmp_dir / "result.glb"

    mesh.export(obj_path)
    mesh.export(glb_path)

    return str(glb_path), str(glb_path), str(obj_path)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="TripoSR 2D to 3D") as demo:
        gr.Markdown(
            f"""
            # TripoSR: 2D 이미지 -> 3D 메시 변환
            - 입력 이미지를 업로드하면 TripoSR로 3D 메시를 생성합니다.
            - 오른쪽 3D 모델 뷰어에서 생성된 GLB 모델을 확인할 수 있습니다.
            - 결과 파일은 `.glb`, `.obj`로 다운로드할 수 있습니다.
            - 실행 디바이스: {get_device_text()}
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="입력 이미지")
                remove_bg = gr.Checkbox(value=True, label="배경 제거")
                foreground_ratio = gr.Slider(
                    minimum=0.6,
                    maximum=0.95,
                    value=0.85,
                    step=0.01,
                    label="전경 비율 (배경 제거 사용 시)",
                )
                mesh_resolution = gr.Slider(
                    minimum=128,
                    maximum=512,
                    value=256,
                    step=32,
                    label="메시 해상도",
                )
                orientation = gr.Dropdown(
                    choices=list(ORIENTATION_TRANSFORMS.keys()),
                    value="자동 보정 (세우기)",
                    label="출력 방향",
                )
                generate_btn = gr.Button("3D 생성", variant="primary")

            with gr.Column(scale=1):
                output_model = gr.Model3D(label="3D 모델 뷰어")
                output_glb = gr.File(label="GLB 다운로드")
                output_obj = gr.File(label="OBJ 다운로드")

        generate_btn.click(
            fn=image_to_3d,
            inputs=[
                input_image,
                remove_bg,
                foreground_ratio,
                mesh_resolution,
                orientation,
            ],
            outputs=[output_model, output_glb, output_obj],
        )

    return demo


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app = build_demo()
    app.launch(inbrowser=True)
