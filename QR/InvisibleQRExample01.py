import qrcode
from PIL import Image, ImageEnhance

def generate_invisible_qr(data, background_path, output_path, alpha=0.1):
    # Step 1: Generate QR code
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H
    )
    qr.add_data(data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

    # Step 2: Make QR code transparent
    datas = img_qr.getdata()
    new_data = []
    for item in datas:
        # Make white background fully transparent, black modules have alpha
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((0, 0, 0, int(255 * alpha)))
    img_qr.putdata(new_data)

    # Step 3: Load background and overlay QR
    background = Image.open(background_path).convert("RGBA")
    bg_w, bg_h = background.size
    qr_w, qr_h = img_qr.size

    # Center the QR code
    pos = ((bg_w - qr_w) // 2, (bg_h - qr_h) // 2)
    background.paste(img_qr, pos, img_qr)

    background.save(output_path)

if __name__ == "__main__":
    # Example usage
    generate_invisible_qr(
        data="http://blog.naver.com/knix009",
        background_path="sample01.png",   # Use a light background for best results
        output_path="invisible_qr.png",
        alpha=0.12                          # Lower alpha means more invisible
    )