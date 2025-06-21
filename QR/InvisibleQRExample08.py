import qrcode
from PIL import Image

# 1. sample01.png 이미지 불러오기 (같은 폴더에 sample01.png가 있어야 합니다)
bg = Image.open("sample01.png").convert("RGBA")

# 2. QR 코드 생성
qr = qrcode.QRCode(
    version=2,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data("http://blog.naver.com/knix009")
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA").resize(bg.size)

# 3. QR코드 알파값 조정 (overlay 효과, 희미하게)
alpha = 40  # 0~255, 숫자가 작을수록 더 투명
qr_mask = qr_img.split()[-1].point(lambda x: alpha if x > 0 else 0)

# 4. overlay 합성
stealth_qr = Image.composite(qr_img, bg, qr_mask)

# 5. 저장
stealth_qr.save("sample01_with_overlay_qr.png")
print("sample01_with_overlay_qr.png 파일이 생성되었습니다.")