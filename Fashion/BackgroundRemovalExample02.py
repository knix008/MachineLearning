import gradio as gr
from PIL import Image
from rembg import remove

def remove_bg_and_set_color_rembg(input_img, bg_color="#FFFFFF"):
    # Remove background using rembg
    img_no_bg = remove(input_img)
    img_no_bg = img_no_bg.convert("RGBA")

    # Create a background image with the desired color
    bg = Image.new("RGBA", img_no_bg.size, bg_color)
    # Composite the foreground with the new background
    out = Image.alpha_composite(bg, img_no_bg)
    return out.convert("RGB")

with gr.Blocks() as demo:
    gr.Markdown("# rembg로 배경 제거 및 바탕색 변경")
    with gr.Row():
        inp = gr.Image(type="pil", label="이미지 업로드")
        color = gr.ColorPicker(label="바탕색", value="#FFFFFF")
    out = gr.Image(type="pil", label="결과 이미지")
    btn = gr.Button("변환하기")

    btn.click(fn=remove_bg_and_set_color_rembg, inputs=[inp, color], outputs=out)

if __name__ == "__main__":
    demo.launch()