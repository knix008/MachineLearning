import os
import re
import tempfile
from typing import List

import gradio as gr
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6 8.0 7.5 7.0 6.1 6.0 5.2 5.0"  # CUDA 아키텍처 설정

def num_range(s: str) -> List[int]:
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [int(x) for x in vals]


# network_pkl을 직접 지정
# Download : https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/
NETWORK_PKL_PATH = "ffhq.pkl"  # 여기 경로를 원하는 pkl 파일로 수정하세요.


def stylemix(row_seeds, col_seeds, style_layers, truncation_psi, noise_mode):
    row_seeds = num_range(row_seeds)
    col_seeds = num_range(col_seeds)
    col_styles = num_range(style_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with dnnlib.util.open_url(NETWORK_PKL_PATH) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    all_images = G.synthesis(all_w, noise_mode=noise_mode)
    all_images = (
        (all_images.permute(0, 2, 3, 1) * 127.5 + 128)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    image_dict = {
        (seed, seed): image for seed, image in zip(all_seeds, list(all_images))
    }

    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (
                (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            )
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()

    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new(
        "RGB", (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), "black"
    )
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(
                PIL.Image.fromarray(image_dict[key], "RGB"), (W * col_idx, H * row_idx)
            )

    tmpdir = tempfile.mkdtemp()
    grid_path = os.path.join(tmpdir, "grid.png")
    canvas.save(grid_path)

    grid_img = PIL.Image.open(grid_path)
    single_images = []
    for (row_seed, col_seed), image in image_dict.items():
        pil_img = PIL.Image.fromarray(image, "RGB")
        single_images.append((f"{row_seed}-{col_seed}.png", pil_img))

    # 10개만 추출
    single_images = [img for _, img in single_images[:10]]
    # 부족한 개수는 None으로 채우기
    while len(single_images) < 10:
        single_images.append(None)

    outputs = [grid_img] + single_images  # 총 11개
    return outputs


with gr.Blocks() as demo:
    gr.Markdown("## StyleGAN2-ADA Style Mixing (Gradio UI)")

    with gr.Row():
        with gr.Column():
            # network_pkl 입력 제거!
            row_seeds = gr.Textbox(label="Row Seeds (예: 85,100,75,458,1500)")
            col_seeds = gr.Textbox(label="Col Seeds (예: 55,821,1789,293)")
            style_layers = gr.Textbox(label="Style Layer Range (예: 0-6)", value="0-6")
            truncation_psi = gr.Slider(
                minimum=0.5, maximum=1.5, value=1.0, step=0.01, label="Truncation Psi"
            )
            noise_mode = gr.Dropdown(
                ["const", "random", "none"], value="const", label="Noise Mode"
            )
            run_btn = gr.Button("이미지 생성")
        with gr.Column():
            out_grid = gr.Image(label="Style Mixing Grid")
            out_img1 = gr.Image(label="이미지 1")
            out_img2 = gr.Image(label="이미지 2")
            out_img3 = gr.Image(label="이미지 3")
            out_img4 = gr.Image(label="이미지 4")
            out_img5 = gr.Image(label="이미지 5")
            out_img6 = gr.Image(label="이미지 6")
            out_img7 = gr.Image(label="이미지 7")
            out_img8 = gr.Image(label="이미지 8")
            out_img9 = gr.Image(label="이미지 9")
            out_img10 = gr.Image(label="이미지 10")

    run_btn.click(
        stylemix,
        [row_seeds, col_seeds, style_layers, truncation_psi, noise_mode],
        [
            out_grid,
            out_img1,
            out_img2,
            out_img3,
            out_img4,
            out_img5,
            out_img6,
            out_img7,
            out_img8,
            out_img9,
            out_img10,
        ],
    )

if __name__ == "__main__":
    demo.launch()
