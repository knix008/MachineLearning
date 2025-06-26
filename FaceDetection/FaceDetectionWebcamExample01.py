import gradio as gr
import cv2
import numpy as np

# 1. 사전 학습된 얼굴 검출용 Haar Cascade 모델을 로드합니다.
#    이 파일은 코드와 같은 디렉터리에 있어야 합니다.
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading cascade file: {e}")
    # 파일이 없을 경우를 대비한 간단한 예외 처리
    face_cascade = None

def detect_faces(image):
    """
    입력된 이미지에서 얼굴을 검출하고, 검출된 얼굴 주위에 사각형을 그립니다.

    Args:
        image (np.ndarray): 웹캠에서 입력받은 이미지 (NumPy 배열 형태).

    Returns:
        np.ndarray: 얼굴에 사각형이 그려진 이미지.
    """
    if face_cascade is None:
        # 모델 로드에 실패했을 경우, 원본 이미지만 반환하고 경고 메시지를 추가합니다.
        cv2.putText(image, "Cascade file not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image

    if image is None:
        # 이미지가 없는 경우 처리
        return None

    # 2. 얼굴 검출을 위해 이미지를 흑백(grayscale)으로 변환합니다.
    #    컬러 이미지보다 흑백 이미지에서 검출 속도가 더 빠르고 효율적입니다.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. 흑백 이미지에서 얼굴을 검출합니다.
    #    - scaleFactor: 이미지 피라미드에서 각 단계별 축소 비율. 1.1은 10%씩 줄여나감을 의미.
    #    - minNeighbors: 얼굴로 확정하기 위해 주변에 필요한 최소 사각형 개수.
    #    - minSize: 검출할 얼굴의 최소 크기.
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # 4. 검출된 각 얼굴의 위치에 사각형을 그립니다.
    #    (x, y)는 사각형의 시작점, (w, h)는 너비와 높이입니다.
    for (x, y, w, h) in faces:
        # 원본 컬러 이미지에 (B, G, R) 순서로 빨간색 사각형을 그립니다. 선 두께는 5.
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # 5. 얼굴에 사각형이 그려진 이미지를 반환합니다.
    return image

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(sources="webcam", type="numpy", label="웹캠 입력 (Webcam Input)"),
    outputs=gr.Image(type="numpy", label="얼굴 검출 결과 (Detection Result)"),
    live=True, # 실시간으로 입력을 처리
    title="실시간 웹캠 얼굴 검출기 (Real-time Webcam Face Detector)",
    description="웹캠을 켜고 화면에 얼굴을 비춰보세요. 실시간으로 얼굴을 찾아 표시해줍니다."
)

# 애플리케이션 실행
if __name__ == "__main__":
    iface.launch()