import gradio as gr
import easyocr
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

# --- 1. 초기화 ---

# 한글 폰트 파일 경로 설정
# 스크립트와 동일한 디렉토리에 'font.ttf' 파일이 있다고 가정합니다.
FONT_PATH = 'NanumGothic-Regular.ttf'

# 폰트 파일이 있는지 확인
if not os.path.exists(FONT_PATH):
    print(f"'{FONT_PATH}' 에서 폰트 파일을 찾을 수 없습니다.")
    print("나눔고딕과 같은 한글 TTF 폰트를 다운로드하여 'font.ttf'로 저장해주세요.")
    # 폰트가 없으면 프로그램 종료
    exit()

# EasyOCR 리더 초기화 (한글과 영어를 사용)
print("EasyOCR 모델을 로딩합니다. 시간이 다소 걸릴 수 있습니다...")
reader = easyocr.Reader(['ko', 'en'])
print("EasyOCR 모델 로딩 완료.")

def draw_boxes_on_image(image, results, font_path, font_size=20):
    """
    EasyOCR 결과(바운딩 박스, 텍스트)를 이미지 위에 그립니다.
    OpenCV 이미지를 PIL 이미지로 변환하여 한글을 처리하고 다시 OpenCV 이미지로 변환합니다.
    """
    # 폰트 로드, 크기는 이미지 높이에 비례하여 조절 가능
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"폰트 파일을 로드할 수 없습니다: {font_path}")
        font = ImageFont.load_default()

    # OpenCV 이미지(BGR)를 PIL 이미지(RGB)로 변환
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for (bbox, text, prob) in results:
        # 바운딩 박스 좌표 추출
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # 박스 그리기 (녹색)
        draw.rectangle([top_left, bottom_right], outline="green", width=2)

        # 텍스트 그리기 (빨간색)
        # 텍스트 배경을 추가하여 가독성 높이기
        text_bbox = draw.textbbox(top_left, text, font=font)
        draw.rectangle(text_bbox, fill="white")
        draw.text(top_left, text, font=font, fill="red")

    # PIL 이미지를 다시 OpenCV 이미지(BGR)로 변환
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def scan_document(image):
    """
    이미지를 입력받아 문서 영역 검출, 원근 변환, 텍스트 추출 및 결과 시각화를 수행합니다.
    """
    if image is None:
        return None, "이미지를 업로드해주세요."

    orig_image = image.copy()

    # --- 2. 문서 영역 검출 및 원근 변환 (OpenCV) ---
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is None:
            print("문서 영역을 찾지 못했습니다. 원본 이미지에서 OCR을 수행합니다.")
            processed_image = orig_image
        else:
            points = doc_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]
            rect[2] = points[np.argmax(s)]
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]
            rect[3] = points[np.argmax(diff)]
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            processed_image = cv2.warpPerspective(orig_image, M, (maxWidth, maxHeight))
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {e}")
        processed_image = orig_image

    # --- 3. 텍스트 추출 및 결과 시각화 ---
    try:
        # EasyOCR로 텍스트와 좌표 정보를 함께 읽기 (detail=1 이 기본값)
        results = reader.readtext(processed_image)

        # 추출된 텍스트를 하나의 문자열로 결합
        extracted_texts = [res[1] for res in results]
        full_text = "\n".join(extracted_texts)

        # 이미지에 바운딩 박스와 텍스트 그리기
        # 이미지 크기에 따라 폰트 크기 동적 조절
        font_size = max(15, int(processed_image.shape[0] / 40))
        result_image_with_boxes = draw_boxes_on_image(processed_image, results, FONT_PATH, font_size)

    except Exception as e:
        full_text = f"텍스트 추출 또는 결과 시각화 중 오류가 발생했습니다: {e}"
        result_image_with_boxes = processed_image

    return result_image_with_boxes, full_text

# --- 4. Gradio 인터페이스 생성 ---
iface = gr.Interface(
    fn=scan_document,
    inputs=gr.Image(type="numpy", label="스캔할 문서 이미지 업로드"),
    outputs=[
        gr.Image(type="numpy", label="처리된 이미지 (인식된 텍스트 박스)"),
        gr.Textbox(label="추출된 텍스트 전체", lines=15)
    ],
    title="OCR 문서 스캐너 (feat. 텍스트 박스 시각화)",
    description="한글 문서 사진을 업로드하면, 인식된 텍스트 영역에 박스를 그리고 텍스트를 함께 보여줍니다.",
    flagging_mode="never",
    examples=[
        ["./sample_doc.jpg"] # 예시 이미지가 있다면 경로를 넣어주세요
    ]
)

# Gradio 앱 실행
if __name__ == "__main__":
    iface.launch()