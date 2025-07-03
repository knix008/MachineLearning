import gradio as gr
from PIL import Image
from rembg import remove
import time


def remove_background(image):
    start_time = time.time()

    input_image = image.convert("RGBA")
    result_image = remove(input_image)
    elapsed_time = time.time() - start_time

    return result_image, f"{elapsed_time:.2f}초 소요됨"


interface = gr.Interface(
    fn=remove_background,
    inputs=gr.Image(type="pil", label="📤 이미지 업로드"),
    outputs=[
        gr.Image(type="pil", label="🖼 배경 제거 결과"),
        gr.Text(label="⏱ 처리 시간"),
    ],
    title="🔮 배경 제거기 (미리보기 전용)",
    description="이미지를 업로드하면 배경이 제거된 결과만 보여줍니다. 저장 기능은 포함되어 있지 않습니다.",
)

interface.launch()
