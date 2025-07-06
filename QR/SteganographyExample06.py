import gradio as gr
from PIL import Image
import numpy as np
import cv2


def bitlist_to_qr(bitlist, size):
    """비트 리스트를 QR코드 이미지로 복원한다."""
    arr = np.array([255 if bit == 0 else 0 for bit in bitlist], dtype=np.uint8)
    arr = arr.reshape(size)
    img = Image.fromarray(arr, mode="L")
    img = img.convert("1")
    return img


def decode_qr_from_image_pil(image):
    """PIL 이미지에서 QR코드 이미지를 추출하고, 텍스트도 반환"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    # 1. 이미지 끝에서 QR크기 추출
    size_bits = []
    for idx in range(2):
        val = 0
        for bit in range(16):
            pixel = image.getpixel((w - 1, h - 1 - idx * 16 - bit))
            val = (val << 1) | (pixel[0] & 1)
        size_bits.append(val)
    qr_w, qr_h = size_bits
    qr_len = qr_w * qr_h
    # 2. QR 비트 추출
    qr_bits = []
    imgdata = iter(image.getdata())
    for i in range(qr_len):
        pixel = next(imgdata)
        qr_bits.append(pixel[0] & 1)
    # 3. QR 이미지로 복원
    qr_img = bitlist_to_qr(qr_bits, (qr_w, qr_h))
    return qr_img


def decode_and_read(image):
    """Gradio용 함수. 입력 이미지를 받아 QR코드 이미지와 추출 텍스트 반환."""
    qr_img = decode_qr_from_image_pil(image)
    # QR 코드 이미지로부터 텍스트 추출 (OpenCV 사용)
    qr_np = np.array(qr_img.convert("L"))
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(qr_np)
    if data:
        qr_text = data
    else:
        qr_text = "QR 코드에서 텍스트를 추출할 수 없습니다."
    return qr_img, qr_text


# The code snippet you provided is creating a Gradio interface for a Steganography QR Code Extractor
# application.
with gr.Blocks() as demo:
    gr.Markdown("# Steganography QR Code Extractor")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="입력 이미지 (QR 코드가 숨김)")
            submit_btn = gr.Button("QR 코드 추출")
        with gr.Column():
            output_image = gr.Image(type="pil", label="복원된 QR 코드 이미지")
            output_text = gr.Textbox(label="QR 코드에서 추출한 텍스트")
    submit_btn.click(
        decode_and_read, inputs=input_image, outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()
