import qrcode
from PIL import Image, ImageEnhance

# 배경 이미지 불러오기
bg = Image.open("sample01.png").convert("RGBA")

# QR 코드 생성
qr = qrcode.QRCode(
    version=2,
    box_size=10,
    border=4,
    error_correction=qrcode.constants.ERROR_CORRECT_H
)
qr.add_data("http://blog.naver.com/knix009")
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA").resize(bg.size)

# QR 코드 알파값 낮추기 (희미하게)
alpha = 40  # 0~255, 숫자가 작을수록 더 투명
qr_mask = qr_img.split()[-1].point(lambda x: alpha if x > 0 else 0)
stealth_qr = Image.composite(qr_img, bg, qr_mask)

# 저장
stealth_qr.save("stealth_qr.png")