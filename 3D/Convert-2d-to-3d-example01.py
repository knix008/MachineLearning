"""
2D RGB 이미지 → 단안 깊이(DPT) → 깊이 맵 이미지, 점군(.ply), 높이장 메시(.obj).

지원 입력 형식: JPG/JPEG, PNG, GIF(첫 프레임), WebP(첫 프레임·정적).

의존성: pip install torch transformers pillow numpy gradio

CLI:
  python Convert-2d-to-3d-example01.py path/to/photo.jpg -o ./image_to_3d_out

Gradio:
  python Convert-2d-to-3d-example01.py --gradio
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from PIL import Image

# Lazy-loaded globals
_processor = None
_model = None
_device = None

ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})


def select_torch_device():
    """NVIDIA CUDA → Apple MPS → CPU 순으로 가속기 선택."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_auto_device_summary() -> str:
    """모델 로드 없이 PyTorch만으로 자동 선택 결과를 문자열로 반환."""
    import torch

    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        return f"자동: CUDA GPU — {torch.cuda.get_device_name(i)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "자동: Apple MPS GPU"
    return "자동: CPU (사용 가능한 CUDA/MPS 없음)"


def get_depth_model():
    global _processor, _model, _device
    if _model is not None:
        return _processor, _model, _device

    import torch
    from transformers import DPTForDepthEstimation, DPTImageProcessor

    _device = select_torch_device()
    _processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    _model = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas",
        low_cpu_mem_usage=True,
    )
    _model.to(_device)
    _model.eval()
    print(
        f"[Convert-2d-to-3d-example01] 깊이 모델 로드: "
        f"{format_auto_device_summary()} (torch.device={_device})"
    )
    return _processor, _model, _device


def load_image_file(path: str) -> Image.Image:
    """
    Load a raster image from disk. GIF / animated WebP: first frame only.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path or "")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"지원하지 않는 확장자입니다: {ext!r}. "
            f"허용: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        return im.convert("RGB")


def gradio_file_to_path(file_value) -> str | None:
    """Normalize Gradio File output to a filesystem path."""
    if file_value is None:
        return None
    if isinstance(file_value, str):
        return file_value if file_value.strip() else None
    if isinstance(file_value, dict):
        for key in ("name", "path"):
            p = file_value.get(key)
            if p:
                return str(p)
        return None
    p = getattr(file_value, "name", None) or getattr(file_value, "path", None)
    return str(p) if p else None


def estimate_depth(image: Image.Image) -> np.ndarray:
    """Return HxW float depth (model units), same spatial size as input image."""
    import torch

    processor, model, device = get_depth_model()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    h, w = image.size[1], image.size[0]
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    return prediction.cpu().numpy().astype(np.float64)


def depth_to_z_for_view(depth: np.ndarray) -> np.ndarray:
    """Invert depth so nearer surfaces get larger Z (camera looks toward +Z)."""
    d = depth.astype(np.float64)
    d = d - d.min()
    d = d / (d.max() - d.min() + 1e-8)
    return 1.0 - d


def unproject_to_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    sample_step: int = 2,
    focal_scale: float = 0.9,
    z_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pinhole-style unprojection. rgb: HxWx3 uint8.
    Returns (N,3) xyz, (N,3) uint8 colors.
    """
    h, w = depth.shape
    z_rel = depth_to_z_for_view(depth) * z_scale
    fx = fy = focal_scale * float(w)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    uu, vv = np.meshgrid(
        np.arange(0, w, sample_step, dtype=np.float64),
        np.arange(0, h, sample_step, dtype=np.float64),
    )
    zz = z_rel[vv.astype(int), uu.astype(int)]
    xx = (uu - cx) * zz / fx
    yy = -(vv - cy) * zz / fy

    colors = rgb[vv.astype(int), uu.astype(int)]
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    cols = colors.reshape(-1, 3)
    return pts, cols


def depth_to_colormap_image(depth: np.ndarray) -> Image.Image:
    """HxW float -> RGB uint8 visualization (magma-like via simple LUT)."""
    d = depth.astype(np.float64)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d = (np.clip(d, 0.0, 1.0) * 255).astype(np.uint8)
    # grayscale to RGB heat tint
    r = np.clip(1.2 * d, 0, 255).astype(np.uint8)
    g = np.clip(0.4 * d + 0.3 * 255 * (1 - d / 255.0), 0, 255).astype(np.uint8)
    b = np.clip(255 - 0.8 * d, 0, 255).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def write_ply_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = xyz.shape[0]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def write_heightfield_obj(
    depth_small: np.ndarray,
    rgb_small: np.ndarray,
    out_path: str,
    z_scale: float = 2.0,
    max_edge_ratio: float = 0.12,
) -> None:
    """
    Simple grid mesh: depth_small HxW, rgb_small HxWx3.
    Skips faces with large depth jump relative to local scale.
    """
    h, w = depth_small.shape
    z_rel = depth_to_z_for_view(depth_small) * z_scale
    fx = fy = 0.9 * float(w)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    verts: list[tuple[float, float, float]] = []
    vmap: dict[tuple[int, int], int] = {}

    def vid(i: int, j: int) -> int:
        key = (i, j)
        if key not in vmap:
            u, v = float(j), float(i)
            zz = float(z_rel[i, j])
            x = (u - cx) * zz / fx
            y = -(v - cy) * zz / fy
            vmap[key] = len(verts)
            verts.append((x, y, zz))
        return vmap[key]

    for i in range(h):
        for j in range(w):
            vid(i, j)

    faces: list[tuple[int, int, int]] = []
    z_span = float(z_rel.max() - z_rel.min() + 1e-6)
    thresh = max_edge_ratio * z_span

    def ok(a: float, b: float, c: float, d: float) -> bool:
        return (
            abs(a - b) < thresh
            and abs(b - c) < thresh
            and abs(c - d) < thresh
            and abs(a - d) < thresh
        )

    for i in range(h - 1):
        for j in range(w - 1):
            z00, z10, z01, z11 = (
                z_rel[i, j],
                z_rel[i, j + 1],
                z_rel[i + 1, j],
                z_rel[i + 1, j + 1],
            )
            if ok(z00, z10, z11, z01):
                v0, v1, v2, v3 = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
                faces.append((v0, v1, v2))
                faces.append((v0, v2, v3))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# heightfield from monocular depth (geometry only; use PLY for RGB)\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")


def resize_pair(
    image: Image.Image, depth: np.ndarray, max_side: int
) -> tuple[Image.Image, np.ndarray]:
    w, h = image.size
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        rgb = np.array(image)
        return image, depth
    nw, nh = int(w * scale), int(h * scale)
    img_r = image.resize((nw, nh), Image.Resampling.LANCZOS)
    d_img = Image.fromarray(depth.astype(np.float32), mode="F")
    d_r = np.array(d_img.resize((nw, nh), Image.Resampling.BICUBIC))
    return img_r, d_r


def process_image(
    image: Image.Image,
    out_dir: str,
    sample_step: int = 2,
    mesh_max_side: int = 256,
    z_scale: float = 2.0,
    focal_scale: float = 0.9,
    basename: str | None = None,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = basename or stamp

    depth = estimate_depth(image)
    depth_vis = depth_to_colormap_image(depth)
    depth_path = os.path.join(out_dir, f"{base}_depth.png")
    depth_vis.save(depth_path)

    rgb = np.array(image.convert("RGB"))
    pts, cols = unproject_to_points(
        depth, rgb, sample_step=sample_step, focal_scale=focal_scale, z_scale=z_scale
    )
    ply_path = os.path.join(out_dir, f"{base}_points.ply")
    write_ply_ascii(ply_path, pts, cols)

    img_m, depth_m = resize_pair(image, depth, mesh_max_side)
    rgb_m = np.array(img_m.convert("RGB"))
    obj_path = os.path.join(out_dir, f"{base}_mesh.obj")
    write_heightfield_obj(depth_m, rgb_m, obj_path, z_scale=z_scale)

    return {
        "depth_png": depth_path,
        "ply": ply_path,
        "obj": obj_path,
    }


def run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="2D image → depth map, PLY point cloud, OBJ heightfield mesh."
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=f"Input image path ({', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))})",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="image_to_3d_out",
        help="Output directory (default: image_to_3d_out)",
    )
    parser.add_argument("--sample-step", type=int, default=2, help="Point cloud stride")
    parser.add_argument(
        "--mesh-max-side",
        type=int,
        default=256,
        help="Longest side (px) for OBJ grid",
    )
    parser.add_argument("--z-scale", type=float, default=2.0, help="Depth extrusion scale")
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio UI instead of CLI",
    )
    args = parser.parse_args()

    if args.gradio:
        return launch_gradio()

    if not args.input:
        parser.print_help()
        print("\nError: input image path is required unless --gradio.", file=sys.stderr)
        return 2

    ext = os.path.splitext(args.input)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        print(
            f"Error: unsupported extension {ext!r}. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
            file=sys.stderr,
        )
        return 2

    image = load_image_file(args.input)
    print(format_auto_device_summary())
    base = os.path.splitext(os.path.basename(args.input))[0]
    paths = process_image(
        image,
        args.output_dir,
        sample_step=args.sample_step,
        mesh_max_side=args.mesh_max_side,
        z_scale=args.z_scale,
        basename=base,
    )
    print("Done:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return 0


def launch_gradio() -> int:
    import gradio as gr

    file_types = [".jpg", ".jpeg", ".png", ".gif", ".webp"]

    def on_run(uploaded, step, mesh_side, z_sc):
        path = gradio_file_to_path(uploaded)
        if not path:
            return (
                None,
                None,
                None,
                None,
                None,
                "JPG, PNG, GIF, WebP 이미지 파일을 선택한 뒤 「변환」을 눌러 주세요.",
            )
        try:
            pil = load_image_file(path)
        except (OSError, ValueError, FileNotFoundError) as e:
            return None, None, None, None, None, f"파일을 읽을 수 없습니다: {e}"

        out_dir = os.path.join(os.path.dirname(__file__), "image_to_3d_out")
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = f"gradio_{stamp}"
        paths = process_image(
            pil,
            out_dir,
            sample_step=int(step),
            mesh_max_side=int(mesh_side),
            z_scale=float(z_sc),
            basename=base,
        )
        depth_img = Image.open(paths["depth_png"])
        msg_lines = [
            f"입력 파일: {path}",
            *(f"{k}: {v}" for k, v in paths.items()),
        ]
        msg = "\n".join(msg_lines)
        obj_path = paths["obj"]
        return pil, depth_img, obj_path, paths["ply"], obj_path, msg

    with gr.Blocks(title="2D → 3D (depth)") as demo:
        gr.Markdown(
            "## 2D 이미지 파일 → 3D (단안 깊이 DPT)\n"
            "**입력:** JPG, JPEG, PNG, GIF, WebP 파일. GIF·애니 WebP는 **첫 프레임**만 사용합니다.\n\n"
            "**출력:** 깊이 맵 이미지, OBJ 메시(3D 뷰어), 점군 PLY 다운로드. 파일은 `3D/image_to_3d_out`에 저장됩니다."
        )
        gr.Textbox(
            label="실행 장치 (자동 감지: CUDA → MPS → CPU)",
            value=format_auto_device_summary(),
            interactive=False,
            lines=2,
        )
        inp = gr.File(
            label="1) 이미지 파일 선택",
            file_count="single",
            file_types=file_types,
        )
        step = gr.Slider(1, 8, value=2, step=1, label="점군 샘플 간격 (클수록 가벼움)")
        mesh_side = gr.Slider(128, 512, value=256, step=32, label="메시용 최대 변 길이")
        z_sc = gr.Slider(0.5, 5.0, value=2.0, step=0.1, label="깊이 돌출 스케일")
        btn = gr.Button("변환", variant="primary")

        gr.Markdown("### 입력 · 출력 (「변환」 후 갱신)")
        with gr.Row(equal_height=True):
            preview_in = gr.Image(
                label="입력 (로드된 RGB)",
                type="pil",
                interactive=False,
                height=420,
            )
            preview_out = gr.Image(
                label="출력: 깊이 맵 (2D)",
                type="pil",
                interactive=False,
                height=420,
            )

        gr.Markdown("### 출력: 3D 메시 미리보기 (OBJ, Gradio 4+ `Model3D`)")
        mesh_view = gr.Model3D(
            label="높이장 메시 (마우스로 회전·확대)",
            height=420,
        )

        with gr.Row():
            ply_f = gr.File(label="점군 PLY 다운로드")
            obj_f = gr.File(label="메시 OBJ 다운로드")
        log = gr.Textbox(label="출력 경로", lines=6)

        out_list = [preview_in, preview_out, mesh_view, ply_f, obj_f, log]
        btn.click(
            on_run,
            inputs=[inp, step, mesh_side, z_sc],
            outputs=out_list,
        )

    demo.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
