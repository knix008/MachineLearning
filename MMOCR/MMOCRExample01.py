import gradio as gr
from mmocr.utils.ocr import MMOCR
from PIL import Image
import numpy as np

# MMOCR 객체 생성 (한글 인식용)
ocr = MMOCR(
    det='DB_r18',
    rec='SAR',
    rec_config='configs/textrecog/sar/sar_r31_parallel_decoder_korean.py',
    rec_ckpt='https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_korean_20211226-1b118d1b.pth'
)

def ocr_infer(image):
    # PIL 이미지를 numpy로 변환 (MMOCR는 numpy 이미지 사용)
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    results = ocr.readtext(image_np, print_result=False)
    # 텍스트 추출
    texts = []
    for res in results:
        for det in res['text']:
            texts.append(det)
    return "\n".join(texts)

with gr.Blocks() as demo:
    gr.Markdown("## MMOCR 한글 인식 데모")
    with gr.Row():
        img_input = gr.Image(type="pil", label="이미지 업로드")
        txt_output = gr.Textbox(label="인식된 한글 텍스트")
    btn = gr.Button("한글 인식 실행")
    btn.click(ocr_infer, inputs=img_input, outputs=txt_output)

if __name__ == "__main__":
    demo.launch()