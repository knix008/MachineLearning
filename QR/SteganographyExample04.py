from PIL import Image
import numpy as np

OUTPUT_IMAGE = "steganography_result.png"     # QR코드가 숨겨진 결과 이미지
QR_IMAGE_EXTRACTED = "extracted_qr.png"       # 복원한 QR코드 이미지


def bitlist_to_qr(bitlist, size):
    """비트 리스트를 QR코드 이미지로 복원한다."""
    arr = np.array([255 if bit == 0 else 0 for bit in bitlist], dtype=np.uint8)
    arr = arr.reshape(size)
    img = Image.fromarray(arr, mode="L")
    # 흑백(1비트)로 변환
    img = img.convert("1")
    return img


def decode_qr_from_image(stego_img_path, qr_img_path):
    image = Image.open(stego_img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    imgdata = iter(image.getdata())
    # 1. 이미지 끝에서 QR크기 추출
    size_bits = []
    for idx in range(2):
        val = 0
        for bit in range(16):
            pixel = image.getpixel((w-1, h-1-idx*16-bit))
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
    qr_img.save(qr_img_path)
    print(f"QR 코드가 {qr_img_path}로 복원되었습니다.")

if __name__ == "__main__":
    print("> Steganography QR Code Exraction Example")
    decode_qr_from_image(OUTPUT_IMAGE, QR_IMAGE_EXTRACTED)
