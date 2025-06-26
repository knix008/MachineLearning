import cv2
import numpy as np
import gradio as gr

# 훈련된 LBPH 인식기와 Haar Cascade 분류기 로드
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except cv2.error as e:
    print("[ERROR] 모델 또는 Cascade 파일을 로드할 수 없습니다.")
    print("[INFO] 'face_training.py'를 먼저 실행하여 'trainer/trainer.yml' 파일을 생성했는지 확인하세요.")
    # Gradio에서 오류를 명확히 표시하기 위해 앱 실행을 중단합니다.
    exit()


# 인식할 사람의 이름 리스트 (훈련 시 사용된 ID 순서와 일치해야 함)
names = ['None', 'Hans Albert Einstein', 'Unknown'] # 예시 이름, 실제 데이터에 맞게 수정하세요.

font = cv2.FONT_HERSHEY_SIMPLEX

def authenticate_face(image):
    """
    입력 이미지에서 얼굴을 감지하고 등록된 사용자인지 인증합니다.
    
    :param image: Gradio 입력을 통해 받은 NumPy 배열 형태의 이미지
    :return: 
        - status (str): 인증 결과 텍스트 ("인증되었습니다", "인증되지 않음")
        - result_image (np.array): 얼굴에 사각형과 이름이 그려진 이미지
    """
    # 기본 인증 상태를 '인증되지 않음'으로 설정
    authentication_status = "인증되지 않음 (Not Authenticated)"
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result_image = image.copy()
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 얼굴이 감지되지 않은 경우
    if len(faces) == 0:
        return "얼굴을 찾을 수 없습니다 (No face detected)", result_image

    # 감지된 모든 얼굴을 순회
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id, confidence = recognizer.predict(roi_gray)

        # 신뢰도 임계값. 이 값은 실험을 통해 조정하는 것이 좋습니다.
        # LBPH의 경우, confidence는 '차이'를 의미하므로 낮을수록 더 확실한 매칭입니다.
        if confidence < 70:
            # 등록된 사용자를 찾았으므로 상태를 '인증'으로 변경하고 루프를 빠져나와도 됩니다.
            # 여기서는 모든 얼굴에 정보를 표시하기 위해 계속 진행합니다.
            authentication_status = "인증되었습니다 (Authenticated)"
            predicted_name = names[id]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            predicted_name = "Unknown"
            confidence_text = ""
        
        # 얼굴 영역에 사각형 그리기
        # 인증 상태에 따라 사각형 색 변경: 인증(녹색), 미인증(빨간색)
        color = (0, 255, 0) if authentication_status == "인증되었습니다 (Authenticated)" else (255, 0, 0)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        
        # 텍스트 표시
        cv2.putText(result_image, predicted_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        if confidence_text:
            cv2.putText(result_image, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    return authentication_status, result_image

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=authenticate_face,
    inputs=gr.Image(type="numpy", label="인증할 이미지를 업로드하세요."),
    # [핵심 변경] 출력을 텍스트박스와 이미지 두 개로 지정
    outputs=[
        gr.Textbox(label="인증 결과 (Authentication Result)"),
        gr.Image(type="numpy", label="처리 결과 (Processing Result)")
    ],
    title="얼굴 인증 시스템",
    description="이미지를 업로드하면 등록된 사용자인지 확인합니다. 등록된 사용자가 한 명이라도 있으면 '인증'으로 표시됩니다."
)

# 앱 실행
iface.launch()