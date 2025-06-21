import qrcode
from PIL import Image
import numpy as np

def set_lsb(value, bit):
    """value의 마지막 비트를 bit로 바꿈"""
    return (value & ~1) | bit

# 1. sample01.png 이미지 불러오기 (RGB로 변환)
bg = Image.open("sample01.png").convert("RGB")
bg_np = np.array(bg)

# 2. QR 코드 생성 (배경보다 작게 생성, 중앙 배치)
qr = qrcode.QRCode(
    version=2,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=6,
    border=4,
)
qr.add_data("http://blog.naver.com/knix009")
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("1")  # 1비트 픽셀
qr_np = np.array(qr_img)
h, w = qr_np.shape

# 3. QR코드를 중앙에 삽입
y0 = (bg_np.shape[0] - h) // 2
x0 = (bg_np.shape[1] - w) // 2

# 4. LSB(가장 마지막 비트) 변경해서 QR 코드 숨기기 (R채널에 삽입)
for y in range(h):
    for x in range(w):
        bit = 0 if qr_np[y, x] == 255 else 1  # 검은색이면 1, 흰색이면 0
        r, g, b = bg_np[y0 + y, x0 + x]
        bg_np[y0 + y, x0 + x, 0] = set_lsb(r, bit)

# 5. 결과 저장
result = Image.fromarray(bg_np)
result.save("sample01_stegano_qr.png")
print("sample01_stegano_qr.png 파일이 생성되었습니다.")