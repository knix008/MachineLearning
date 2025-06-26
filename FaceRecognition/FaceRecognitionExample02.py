import cv2
import numpy as np
import gradio as gr
from PIL import ImageFont, ImageDraw, Image
import sys

# 한글 폰트 파일 경로 (반드시 프로젝트 디렉터리에 .ttf 파일이 필요합니다)
FONT_PATH = "NanumGothic-Regular.ttf"  # 적절한 한글 폰트 파일로 수정하세요
FONT_SIZE = 32

# 훈련된 LBPH 인식기와 Haar Cascade 분류기 로드
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise cv2.error("Cascade file could not be loaded.")
except cv2.error as e:
    print("[ERROR] 모델 또는 Cascade 파일을 로드할 수 없습니다.")
    print("[INFO] 'face_training.py'를 먼저 실행하여 'trainer/trainer.yml' 파일을 생성했는지 확인하세요.")
    sys.exit()

# 인식할 사람의 이름 리스트 (훈련 시 사용된 ID 순서와 일치해야 함)
names = ['None', '아인슈타인', '모르는 사람'] # 실제 데이터에 맞게 수정

def put_text_korean(img, text, org, font_path=FONT_PATH, font_size=FONT_SIZE, color=(255,255,255)):
    """이미지에 한글(및 영문) 텍스트를 씁니다."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()
    # color: BGR → RGB
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def authenticate_face(image):
    authentication_status = "인증되지 않음 (Not Authenticated)"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Pillow와 호환을 위해 BGR로 변환
    result_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        # Gradio 출력은 RGB여야 함
        return "얼굴을 찾을 수 없습니다 (No face detected)", cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id, confidence = recognizer.predict(roi_gray)

        if confidence < 70 and 0 <= id < len(names):
            authentication_status = "인증되었습니다 (Authenticated)"
            predicted_name = names[id]
            confidence_text = f"{round(100 - confidence)}%"
            color = (0, 255, 0)
        else:
            predicted_name = "Unknown"
            confidence_text = ""
            color = (255, 0, 0)

        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        # 한글 이름
        result_image = put_text_korean(
            result_image, predicted_name, (x + 5, y - 35), font_size=FONT_SIZE, color=color
        )
        # 신뢰도 (숫자, 한글 아님)
        if confidence_text:
            result_image = put_text_korean(
                result_image, confidence_text, (x + 5, y + h - 5), font_size=FONT_SIZE-4, color=(255,255,0)
            )

    # Gradio 출력은 RGB여야 함
    return authentication_status, cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

iface = gr.Interface(
    fn=authenticate_face,
    inputs=gr.Image(type="numpy", label="인증할 이미지를 업로드하세요."),
    outputs=[
        gr.Textbox(label="인증 결과 (Authentication Result)"),
        gr.Image(type="numpy", label="처리 결과 (Processing Result)")
    ],
    title="얼굴 인증 시스템",
    description="이미지를 업로드하면 등록된 사용자인지 확인합니다. 등록된 사용자가 한 명이라도 있으면 '인증'으로 표시됩니다."
)

iface.launch()