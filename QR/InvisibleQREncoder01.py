import gradio as gr
import numpy as np
from PIL import Image
import qrcode
import cv2
import os

def generate_qr_code(data, size=256):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=1,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("L")
    img_qr = img_qr.resize((size, size), Image.NEAREST)
    return img_qr

def embed_qr_in_image(cover_img, qr_data):
    cover_img = cover_img.convert("RGB")
    cover = np.array(cover_img)
    h, w, ch = cover.shape

    qr_img = generate_qr_code(qr_data, size=min(h, w))
    qr = np.array(qr_img)
    qr = (qr > 128).astype(np.uint8)

    qr = cv2.resize(qr, (min(h, w), min(h, w)), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:qr.shape[0], :qr.shape[1]] = qr

    stego = cover.copy()
    stego[..., 2] = (stego[..., 2] & np.uint8(0xFE)) | mask

    stego_img = Image.fromarray(stego)
    output_path = os.path.abspath('stego_output.png')
    stego_img.save(output_path)
    # QR 이미지도 PIL 이미지로 반환
    return output_path, qr_img

def extract_qr_from_image_file(stego_img):
    # stego_img는 PIL.Image
    stego_img = stego_img.convert("RGB")
    stego = np.array(stego_img)
    h, w, ch = stego.shape

    qr_mask = (stego[..., 2] & 1).astype(np.uint8) * 255

    qr_mask_img = Image.fromarray(qr_mask).convert("L")
    qr_mask_np = np.array(qr_mask_img)
    detector = cv2.QRCodeDetector()
    data, pts, _ = detector.detectAndDecode(qr_mask_np)

    qr_mask_path = os.path.abspath('extracted_qr.png')
    qr_mask_img.save(qr_mask_path)

    if data:
        return stego_img, qr_mask_path, qr_mask_img, f"Decoded QR data: {data}"
    else:
        return stego_img, qr_mask_path, qr_mask_img, "QR code could not be decoded. Try a different image or adjust parameters."

with gr.Blocks() as demo:
    gr.Markdown("# Invisible QR Code Steganography Demo")
    with gr.Tab("Embed QR"):
        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(label="Cover Image", type="pil")
                qr_text = gr.Textbox(label="QR Code Data (Text/URL)", value="https://blog.naver.com/knix009")
                stego_btn = gr.Button("Embed QR Code")
            with gr.Column():
                stego_file = gr.File(label="Download Image with Invisible QR")
                qr_preview = gr.Image(label="QR Code Preview", type="pil")
        stego_btn.click(
            fn=embed_qr_in_image,
            inputs=[inp_img, qr_text],
            outputs=[stego_file, qr_preview]
        )
    with gr.Tab("Extract QR"):
        with gr.Row():
            with gr.Column():
                extract_file = gr.Image(label="Stego Image File (.png)", type="pil")
                extract_btn = gr.Button("Extract QR Code")
            with gr.Column():
                qr_file = gr.File(label="Download Extracted QR Mask")
                qr_mask_preview = gr.Image(label="Extracted QR Mask Preview", type="pil")
                qr_text_out = gr.Textbox(label="Decoded QR Data")
        extract_btn.click(
            fn=extract_qr_from_image_file,
            inputs=extract_file,
            outputs=[extract_file, qr_file, qr_mask_preview, qr_text_out]
        )

if __name__ == "__main__":
    demo.launch()