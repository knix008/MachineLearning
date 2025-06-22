from PIL import Image
import qrcode
import numpy as np

INPUT_IMAGE = "sample01.png"           # 원본 이미지
INPUT_TEXT = "http://blog.naver.com/knix009"  # 숨길 텍스트
OUTPUT_IMAGE = "steganography_result.png"     # QR코드가 숨겨진 결과 이미지
QR_IMAGE_INPUT = "qr_code.png"                # QR코드 이미지 (생성용)
QR_IMAGE_EXTRACTED = "extracted_qr.png"       # 복원한 QR코드 이미지

def generate_qr_code(text, qr_size=128):
    """입력 텍스트로 QR코드 이미지를 생성한다."""
    qr = qrcode.QRCode(
        version=1,  # 자동 조정: None으로 해도 됨
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=2,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("1")
    img = img.resize((qr_size, qr_size), Image.NEAREST)
    img.save(QR_IMAGE_INPUT)
    return img

def qr_to_bitlist(qr_img):
    """QR코드 이미지를 1차원 비트 리스트로 변환한다."""
    arr = np.array(qr_img)
    # 0: 검은색(1), 255: 흰색(0)로 변환
    bitlist = [0 if pixel else 1 for row in arr for pixel in row]
    return bitlist, qr_img.size

def bitlist_to_qr(bitlist, size):
    """비트 리스트를 QR코드 이미지로 복원한다."""
    arr = np.array([255 if bit == 0 else 0 for bit in bitlist], dtype=np.uint8)
    arr = arr.reshape(size)
    img = Image.fromarray(arr, mode="L")
    # 흑백(1비트)로 변환
    img = img.convert("1")
    return img

def encode_qr_in_image(input_img_path, text, output_img_path):
    # 1. 입력 텍스트로 QR코드 생성
    qr_img = generate_qr_code(text)
    qr_bits, qr_size = qr_to_bitlist(qr_img)
    qr_len = len(qr_bits)
    
    # 2. 입력 이미지 열기
    image = Image.open(input_img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    newimg = image.copy()
    w, h = newimg.size

    # 3. 스테가노그래피 삽입 (LSB)
    imgdata = iter(newimg.getdata())
    for i in range(qr_len):
        pixel = list(next(imgdata))
        # QR bit를 R채널에 LSB로 숨김
        pixel[0] = (pixel[0] & ~1) | qr_bits[i]
        newimg.putpixel((i % w, i // w), tuple(pixel))
    # QR 이미지 크기 정보(너비, 높이)를 마지막 32픽셀에 인코딩 (16+16비트)
    for idx, val in enumerate([qr_size[0], qr_size[1]]):
        for bit in range(16):
            pixel = list(next(imgdata))
            pixel[0] = (pixel[0] & ~1) | ((val >> (15 - bit)) & 1)
            # 뒤에서부터 32픽셀에 기록 (좌표는 무의미)
            newimg.putpixel((w-1, h-1-idx*16-bit), tuple(pixel))
    newimg.save(output_img_path)
    print(f"QR 코드가 {output_img_path}에 숨겨졌습니다.")

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
    import sys
    print("> Welcome to QR Steganography ::\n1. Encode QR\n2. Decode QR")
    choice = input("> ")
    if choice == "1":
        encode_qr_in_image(INPUT_IMAGE, INPUT_TEXT, OUTPUT_IMAGE)
    elif choice == "2":
        decode_qr_from_image(OUTPUT_IMAGE, QR_IMAGE_EXTRACTED)
    else:
        print("> Invalid choice, exiting.")