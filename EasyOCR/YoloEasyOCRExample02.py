import gradio as gr
import easyocr
import cv2
import numpy as np
import time # 시간 측정을 위해 time 모듈 추가

# --- 1. 초기화 ---

# EasyOCR 리더 초기화 (한글과 영어를 사용)
print("EasyOCR 모델을 로딩합니다. 시간이 다소 걸릴 수 있습니다...")
reader = easyocr.Reader(['ko', 'en'])
print("EasyOCR 모델 로딩 완료.")

def draw_boxes_on_image(image, results):
    """
    OpenCV를 사용하여 이미지 위에 바운딩 박스만 그립니다.
    (텍스트 표시는 제거되었습니다.)
    """
    # 원본 이미지를 수정하지 않기 위해 복사본 생성
    image_with_boxes = image.copy()

    for (bbox, text, prob) in results:
        # 바운딩 박스 좌표를 numpy 배열로 변환
        # cv2.polylines는 정수형 좌표를 받습니다.
        points = np.array(bbox, dtype=np.int32)

        # 이미지 위에 폴리곤(다각형) 형태의 박스 그리기
        cv2.polylines(image_with_boxes, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    return image_with_boxes


def scan_document(image):
    """
    이미지를 입력받아 문서 영역 검출, 원근 변환, 텍스트 추출을 수행하고,
    처리 시간을 측정하여 함께 반환합니다.
    """
    # --- 시간 측정 시작 ---
    start_time = time.time()

    if image is None:
        return None, "이미지를 업로드해주세요.", "처리 시간: 0.00초"

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
        results = reader.readtext(processed_image)

        extracted_texts = [res[1] for res in results]
        full_text = "\n".join(extracted_texts)

        # 이미지에 바운딩 박스만 그리기
        result_image_with_boxes = draw_boxes_on_image(processed_image, results)

    except Exception as e:
        full_text = f"텍스트 추출 또는 결과 시각화 중 오류가 발생했습니다: {e}"
        result_image_with_boxes = processed_image

    # --- 시간 측정 종료 및 결과 문자열 생성 ---
    end_time = time.time()
    processing_time = end_time - start_time
    time_string = f"총 소요 시간: {processing_time:.2f}초"
    print(time_string) # 터미널에도 시간 출력

    # 3개의 결과를 반환
    return result_image_with_boxes, full_text, time_string

# --- 4. Gradio 인터페이스 생성 ---
iface = gr.Interface(
    fn=scan_document,
    inputs=gr.Image(type="numpy", label="스캔할 문서 이미지 업로드"),
    outputs=[
        gr.Image(type="numpy", label="처리된 이미지 (인식된 텍스트 박스)"),
        gr.Textbox(label="추출된 텍스트 전체", lines=15),
        gr.Textbox(label="처리 시간") # 처리 시간을 표시할 출력 컴포넌트 추가
    ],
    title="OCR 문서 스캐너",
    description="한글 문서 사진을 업로드하면, 인식된 텍스트 영역에 박스를 그리고 처리 시간을 알려줍니다.",
    flagging_mode="never",
    examples=[
        ["./sample_doc.jpg"] # 예시 이미지가 있다면 경로를 넣어주세요
    ]
)

# Gradio 앱 실행
if __name__ == "__main__":
    iface.launch()