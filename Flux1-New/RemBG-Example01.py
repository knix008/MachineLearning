import os
import tempfile
from datetime import datetime

import gradio as gr
from PIL import Image
from rembg import remove, new_session

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

def build_filename(model_name, bg_type, bg_color, alpha_matting, alpha_matting_fg, alpha_matting_bg, alpha_matting_erode, only_mask, post_process, ext):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    bg_part = "transparent" if bg_type == "투명" else f"bg{bg_color.lstrip('#')}"

    if alpha_matting:
        am_part = f"am-fg{int(alpha_matting_fg)}-bg{int(alpha_matting_bg)}-er{int(alpha_matting_erode)}"
    else:
        am_part = "am-off"

    flags = []
    if only_mask:
        flags.append("mask")
    if post_process:
        flags.append("pp")
    flags_part = "-".join(flags) if flags else "std"

    name = f"{SCRIPT_NAME}_{now}_{model_name}_{bg_part}_{am_part}_{flags_part}.{ext}"
    return name

def remove_bg_and_set_color_rembg(input_img, bg_type, bg_color, model_name, alpha_matting, alpha_matting_fg, alpha_matting_bg, alpha_matting_erode, only_mask, post_process):
    session = new_session(model_name)
    input_img = Image.open(input_img)
    img_no_bg = remove(
        input_img,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=int(alpha_matting_fg),
        alpha_matting_background_threshold=int(alpha_matting_bg),
        alpha_matting_erode_size=int(alpha_matting_erode),
        only_mask=only_mask,
        post_process_mask=post_process,
    )
    img_no_bg = img_no_bg.convert("RGBA")

    if only_mask:
        result_img = img_no_bg.convert("RGB")
        ext = "jpg"
    elif bg_type == "투명":
        result_img = img_no_bg  # RGBA (투명 배경)
        ext = "png"
    else:
        bg = Image.new("RGBA", img_no_bg.size, bg_color)
        result_img = Image.alpha_composite(bg, img_no_bg).convert("RGB")
        ext = "jpg"

    filename = build_filename(model_name, bg_type, bg_color, alpha_matting, alpha_matting_fg, alpha_matting_bg, alpha_matting_erode, only_mask, post_process, ext)
    save_path = os.path.join(tempfile.gettempdir(), filename)

    save_kwargs = {"quality": 95} if ext == "jpg" else {}
    result_img.save(save_path, **save_kwargs)

    return result_img, gr.DownloadButton(value=save_path, interactive=True)

def toggle_color_picker(bg_type):
    return gr.update(visible=(bg_type == "색상"))

css = """
#download-btn { min-height: unset !important; height: 44px !important; }
#download-btn .wrap { min-height: unset !important; height: 44px !important; padding: 0 16px !important; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# rembg로 배경 제거 및 바탕색 변경")

    with gr.Row():
        inp = gr.Image(type="filepath", label="이미지 업로드", height=800)
        out = gr.Image(type="pil", label="결과 이미지", height=800)

    with gr.Row():
        bg_type = gr.Radio(
            choices=["투명", "색상"],
            value="색상",
            label="배경 유형",
            info="투명: PNG 알파 채널로 출력 / 색상: 지정한 색으로 배경을 채워 JPG로 출력합니다."
        )
        color = gr.ColorPicker(
            label="바탕색",
            value="#FFFFFF",
            visible=True,
            info="배경을 제거한 뒤 채울 색상입니다. '마스크만 출력'이 활성화된 경우 무시됩니다."
        )
        model_name = gr.Dropdown(
            choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "sam"],
            value="u2net",
            label="모델 선택",
            info="배경 제거에 사용할 AI 모델입니다. u2net(범용), u2net_human_seg(인물), isnet-anime(애니/일러스트), sam(고정밀) 등 용도에 맞게 선택하세요."
        )

    bg_type.change(fn=toggle_color_picker, inputs=bg_type, outputs=color)

    with gr.Accordion("고급 설정 (Alpha Matting)", open=False):
        gr.Markdown(
            "> **Alpha Matting**은 머리카락·털·반투명 영역 등 경계가 복잡한 피사체의 가장자리를 더 부드럽고 자연스럽게 처리합니다. "
            "활성화하면 처리 시간이 다소 길어질 수 있습니다."
        )
        alpha_matting = gr.Checkbox(
            label="Alpha Matting 활성화",
            value=False,
            info="켜면 아래 세부 설정이 적용됩니다."
        )
        with gr.Row():
            alpha_matting_fg = gr.Slider(
                0, 255, value=240, step=1,
                label="Foreground Threshold",
                info="이 값 이상의 픽셀은 확실한 전경으로 처리합니다. 값이 높을수록 전경 영역이 좁아집니다. (기본값: 240)"
            )
            alpha_matting_bg = gr.Slider(
                0, 255, value=10, step=1,
                label="Background Threshold",
                info="이 값 이하의 픽셀은 확실한 배경으로 처리합니다. 값이 낮을수록 배경 영역이 넓어집니다. (기본값: 10)"
            )
            alpha_matting_erode = gr.Slider(
                0, 40, value=10, step=1,
                label="Erode Size",
                info="마스크 경계를 침식(축소)하는 크기입니다. 값이 클수록 피사체 가장자리가 더 많이 깎입니다. (기본값: 10)"
            )

    with gr.Row():
        only_mask = gr.Checkbox(
            label="마스크만 출력",
            value=False,
            info="체크하면 배경색 합성 없이 흑백 마스크 이미지만 반환합니다. 마스크를 별도로 활용할 때 사용하세요."
        )
        post_process = gr.Checkbox(
            label="Post Process Mask",
            value=True,
            info="마스크에 후처리를 적용해 노이즈를 줄이고 경계를 더 깔끔하게 정리합니다."
        )

    with gr.Row():
        btn = gr.Button("변환하기", variant="primary")
        download = gr.DownloadButton(label="다운로드", interactive=False, elem_id="download-btn")

    btn.click(
        fn=remove_bg_and_set_color_rembg,
        inputs=[inp, bg_type, color, model_name, alpha_matting, alpha_matting_fg, alpha_matting_bg, alpha_matting_erode, only_mask, post_process],
        outputs=[out, download]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
