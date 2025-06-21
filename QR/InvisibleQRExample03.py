import cv2
import numpy as np
import qrcode

# Parameters
data = "http://blog.naver.com/knix009"  # Data to encode in QR code
qr_size = 200  # Size of QR code
alpha = 0.15   # Transparency of QR overlay

# Generate QR code
qr = qrcode.QRCode(box_size=10, border=1)
qr.add_data(data)
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white").convert('L')
qr_img = qr_img.resize((qr_size, qr_size))
qr_array = np.array(qr_img)

# Load sample image
img = cv2.imread('sample01.png')
if img is None:
    raise FileNotFoundError("sample01.png not found in the current directory.")

# Resize QR to fit on image
h, w, _ = img.shape
x_offset = w - qr_size - 10
y_offset = h - qr_size - 10

# Prepare QR for overlay: make white pixels transparent
qr_rgb = cv2.cvtColor(qr_array, cv2.COLOR_GRAY2BGR)
mask = qr_array < 128  # QR black pixels

# Overlay QR code invisibly (low alpha)
roi = img[y_offset:y_offset+qr_size, x_offset:x_offset+qr_size]
roi[mask] = cv2.addWeighted(roi[mask], 1-alpha, qr_rgb[mask], alpha, 0)

img[y_offset:y_offset+qr_size, x_offset:x_offset+qr_size] = roi

# Save result
cv2.imwrite('invisible_qr_result.jpg', img)
print("Invisible QR code embedded and saved as invisible_qr_result.jpg")