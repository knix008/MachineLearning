import gradio as gr
from PIL import Image
import qrcode
import numpy as np
import cv2


def generate_qr_code(text, qr_size=128):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=2,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("1")
    img = img.resize((qr_size, qr_size), Image.NEAREST)
    return img


def qr_to_bitlist(qr_img):
    arr = np.array(qr_img)
    bitlist = [0 if pixel else 1 for row in arr for pixel in row]
    return bitlist, qr_img.size


def bitlist_to_qr(bitlist, size):
    arr = np.array([255 if bit == 0 else 0 for bit in bitlist], dtype=np.uint8)
    arr = arr.reshape(size)
    img = Image.fromarray(arr, mode="L").convert("1")
    return img


def encode_qr_in_image(input_img, text):
    qr_img = generate_qr_code(text)
    qr_bits, qr_size = qr_to_bitlist(qr_img)
    qr_len = len(qr_bits)
    image = input_img.convert("RGB")
    newimg = image.copy()
    w, h = newimg.size
    if w * h < qr_len + 32:
        raise ValueError("입력 이미지가 너무 작습니다. 더 큰 이미지를 사용하세요.")
    imgdata = iter(newimg.getdata())
    for i in range(qr_len):
        pixel = list(next(imgdata))
        pixel[0] = (pixel[0] & ~1) | qr_bits[i]
        newimg.putpixel((i % w, i // w), tuple(pixel))
    for idx, val in enumerate([qr_size[0], qr_size[1]]):
        for bit in range(16):
            pixel = list(next(imgdata))
            pixel[0] = (pixel[0] & ~1) | ((val >> (15 - bit)) & 1)
            newimg.putpixel((w - 1, h - 1 - idx * 16 - bit), tuple(pixel))
    return qr_img, newimg


def decode_qr_from_image(stego_img):
    image = stego_img.convert("RGB")
    w, h = image.size
    size_bits = []
    for idx in range(2):
        val = 0
        for bit in range(16):
            pixel = image.getpixel((w - 1, h - 1 - idx * 16 - bit))
            val = (val << 1) | (pixel[0] & 1)
        size_bits.append(val)
    qr_w, qr_h = size_bits
    qr_len = qr_w * qr_h
    print(f"QR 코드 크기: {qr_w}x{qr_h}, 비트 길이: {qr_len}")
    imgdata = iter(image.getdata())
    qr_bits = []
    for i in range(qr_len):
        pixel = next(imgdata)
        qr_bits.append(pixel[0] & 1)
    qr_img = bitlist_to_qr(qr_bits, (qr_w, qr_h))
    return qr_img


def read_qr_text(qr_img):
    # pyzbar 대신 OpenCV의 QRCodeDetector 사용
    qr_img_cv = np.array(qr_img.convert("L"))
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(qr_img_cv)
    if data:
        return data
    else:
        return "(QR코드에서 텍스트를 추출하지 못했습니다)"


def process(input_img, qr_text):
    if input_img is None or not qr_text:
        return None, None, None, ""
    qr_img, stego_img = encode_qr_in_image(input_img, qr_text)
    extracted_qr = decode_qr_from_image(stego_img)
    extracted_text = read_qr_text(extracted_qr)
    return qr_img, stego_img, extracted_qr, extracted_text


demo = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="입력 이미지"),
        gr.Textbox(label="QR 코드에 넣을 텍스트 (예: URL 등)"),
    ],
    outputs=[
        gr.Image(type="pil", label="생성된 QR 코드"),
        gr.Image(type="pil", label="QR이 숨겨진 결과 이미지 (Stego)"),
        gr.Image(type="pil", label="Stego 이미지에서 추출한 QR 코드"),
        gr.Textbox(label="QR 코드에서 추출한 텍스트"),
    ],
    title="QR Code Steganography with Gradio",
    description="이미지에 QR 코드를 스테가노그래피로 숨깁니다. 입력 이미지를 업로드하고 QR 코드에 넣을 텍스트를 입력하세요. Stego 이미지에서 QR을 추출하고, 텍스트도 읽어줍니다.",
)

if __name__ == "__main__":
    demo.launch()
