from PIL import Image
import qrcode

# 고정된 파일명
INPUT_IMAGE = "sample01.png"
OUTPUT_IMAGE = "steganography_qr.png"
QR_INPUT_IMAGE = "qr_input.png"
QR_OUTPUT_IMAGE = "qr_output.png"

# 고정된 QR 코드 데이터
QR_TEXT = "http://blog.naver.com/knix009"

def genData(data):
    return [format(ord(i), "08b") for i in data]

def modPix(pix, data):
    datalist = genData(data)
    lendata = len(datalist)
    imdata = iter(pix)
    for i in range(lendata):
        pixels = [value for value in next(imdata)[:3] + next(imdata)[:3] + next(imdata)[:3]]
        for j in range(8):
            if datalist[i][j] == "0" and pixels[j] % 2 != 0:
                pixels[j] -= 1
            elif datalist[i][j] == "1" and pixels[j] % 2 == 0:
                pixels[j] = pixels[j] - 1 if pixels[j] != 0 else pixels[j] + 1
        if i == lendata - 1:
            pixels[-1] |= 1
        else:
            pixels[-1] &= ~1
        yield tuple(pixels[:3])
        yield tuple(pixels[3:6])
        yield tuple(pixels[6:9])

def encode_enc(newimg, data):
    w = newimg.size[0]
    (x, y) = (0, 0)
    for pixel in modPix(newimg.getdata(), data):
        newimg.putpixel((x, y), pixel)
        x = 0 if x == w - 1 else x + 1
        y += 1 if x == 0 else 0

def encode_with_qr():
    # 1. 고정된 텍스트를 QR 코드로 변환
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(QR_TEXT)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_img.save(QR_INPUT_IMAGE)
    # 2. sample01.png에 QR 코드 텍스트를 스테가노그래피로 삽입
    base_img = Image.open(INPUT_IMAGE).convert("RGB")
    newimg = base_img.copy()
    encode_enc(newimg, QR_TEXT)
    newimg.save(OUTPUT_IMAGE)
    print(f"[INFO] Steganography image saved as {OUTPUT_IMAGE}")

def decode_from_qr():
    # 1. steganography_qr.png에서 텍스트 추출
    image = Image.open(OUTPUT_IMAGE, "r")
    imgdata = iter(image.getdata())
    data = ""
    while True:
        pixels = [value for value in next(imgdata)[:3] + next(imgdata)[:3] + next(imgdata)[:3]]
        binstr = "".join(["1" if i % 2 else "0" for i in pixels[:8]])
        data += chr(int(binstr, 2))
        if pixels[-1] % 2 != 0:
            break
    # 2. 텍스트를 QR 코드로 변환하여 저장
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_img.save(QR_OUTPUT_IMAGE)
    print(f"[INFO] Decoded text: {data}")
    print(f"[INFO] QR code image saved as {QR_OUTPUT_IMAGE}")
    return data

def main():
    print("> Steganography with QR Code ::\n1. Encode(text→QR+stegano)\n2. Decode(stegano→text→QR)")
    choice = input("Choice: ")
    if choice == "1":
        encode_with_qr()
    elif choice == "2":
        decode_from_qr()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()