import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # 사전 학습된 YOLOv8 나노 모델 사용

# 성별 분류 모델 로드
gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# 동영상 파일 열기
video_path = '7647783-hd_1920_1080_30fps.mp4'  # 실제 동영상 파일 경로로 변경하세요.
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: 동영상을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8로 객체 탐지
    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 'person' 클래스만 처리
            if model.names[int(box.cls)] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 사람 영역 잘라내기
                person_face = frame[y1:y2, x1:x2]

                if person_face.size == 0:
                    continue

                # 성별 분류를 위한 전처리
                blob = cv2.dnn.blobFromImage(person_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                # 성별 예측
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # 결과 시각화
                label = f'Person: {gender}'
                color = (0, 255, 0) if gender == 'Male' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 결과 프레임 보여주기
    cv2.imshow('Gender Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()