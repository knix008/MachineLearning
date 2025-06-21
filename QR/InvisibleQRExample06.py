import qrcode
from PIL import Image
import numpy as np

bg = Image.open("sample01.png").convert("RGBA")
bg_np = np.array(bg)

qr = qrcode.QRCode(
    version=2,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=8,  # 더 크게
    border=4,
)
qr.add_data("http://blog.naver.com/knix009")
qr.make(fit=True)

qr_img = qr.make_image(fill_color="black", back_color="white").convert("L")
qr_np = np.array(qr_img)
h, w = qr_np.shape

y0 = (bg_np.shape[0] - h) // 2
x0 = (bg_np.shape[1] - w) // 2

noise_strength = 18  # 기존보다 강하게 (10~20 권장)

for y in range(h):
    for x in range(w):
        if qr_np[y, x] < 128:
            for c in range(3):
                orig = bg_np[y0 + y, x0 + x, c]
                new = max(0, min(255, orig - noise_strength))
                bg_np[y0 + y, x0 + x, c] = new

result = Image.fromarray(bg_np)
result.save("sample01_with_noisy_qr_strong.png")
print("sample01_with_noisy_qr_strong.png 파일이 생성되었습니다.")