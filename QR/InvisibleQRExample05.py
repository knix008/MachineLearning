import qrcode
from PIL import Image
import numpy as np

# 1. sample01.png 불러오기 (RGBA로 변환)
bg = Image.open("sample01.png").convert("RGBA")
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

qr_img = qr.make_image(fill_color="black", back_color="white").convert("L")
qr_np = np.array(qr_img)
h, w = qr_np.shape

# 3. QR코드가 들어갈 위치 계산 (중앙)
y0 = (bg_np.shape[0] - h) // 2
x0 = (bg_np.shape[1] - w) // 2

# 4. 배경 이미지의 해당 영역에 QR코드 노이즈 삽입
#    - QR코드 검은 부분이면 픽셀값을 아주 미세하게 감소(어둡게)
#    - 흰 부분이면 그대로
noise_strength = 5  # 픽셀값을 얼마나 조정할지 (1~10 사이 추천, 크면 인식 잘됨, 작으면 더 안 보임)

for y in range(h):
    for x in range(w):
        if qr_np[y, x] < 128:  # QR 검은 점
            for c in range(3):  # R,G,B만
                orig = bg_np[y0 + y, x0 + x, c]
                new = max(0, min(255, orig - noise_strength))
                bg_np[y0 + y, x0 + x, c] = new  # 아주 미세하게 어둡게
        # else: 흰색이면 변화 없음

# 5. 결과 저장 및 미리보기
result = Image.fromarray(bg_np)
result.save("sample01_with_noisy_qr.png")
print("sample01_with_noisy_qr.png 파일이 생성되었습니다.")