import cv2
import numpy as np
from PIL import Image # Pillow 라이브러리 import
import os

# --- 설정 ---
# 학습할 이미지가 있는 폴더 경로
dataset_path = 'dataset'
# 훈련된 모델을 저장할 파일 경로
trainer_file = 'trainer/trainer.yml'

# 'trainer' 폴더가 없으면 생성
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# --- 얼굴 인식기 및 감지기 준비 ---
# LBPH 얼굴 인식기 생성
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Haar Cascade 얼굴 감지기 로드
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    """
    dataset 폴더에서 모든 이미지와 각 이미지에 해당하는 ID(레이블)를 가져오는 함수
    """
    # 폴더 내의 모든 이미지 파일 경로를 리스트로 만듦
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
    face_samples = []
    ids = []

    print("이미지를 불러오는 중입니다...")
    
    for image_path in image_paths:
        # 이미지를 열고 흑백(grayscale)으로 변환
        # 흑백 이미지가 색상 정보가 없어 특징을 더 간단하고 명확하게 분석할 수 있음
        pil_image = Image.open(image_path).convert('L')
        img_numpy = np.array(pil_image, 'uint8')

        # 파일 이름에서 ID 번호(레이블) 추출
        try:
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
        except ValueError:
            print(f"파일 이름 형식이 잘못되었습니다: {image_path}. 'User.ID.SampleID.jpg' 형식이어야 합니다.")
            continue

        # 이미지에서 얼굴 감지
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            # 감지된 얼굴 부분만 잘라내어 리스트에 추가
            face_samples.append(img_numpy[y:y+h, x:x+w])
            # 해당 얼굴의 ID를 리스트에 추가
            ids.append(face_id)
            
    print(f"{len(face_samples)}개의 얼굴 이미지를 성공적으로 불러왔습니다.")
    return face_samples, ids

# --- 메인 실행 부분 ---
print("\n[INFO] 얼굴 데이터로 모델을 훈련합니다. 잠시만 기다려 주세요...")

# 이미지와 레이블 불러오기
faces, ids = get_images_and_labels(dataset_path)

# 불러온 얼굴 데이터와 ID로 얼굴 인식기 훈련
recognizer.train(faces, np.array(ids))

# 훈련된 결과를 파일로 저장
recognizer.write(trainer_file)

# 훈련된 사람(ID) 수와 프로그램 종료 메시지 출력
print(f"\n[SUCCESS] {len(np.unique(ids))}명의 얼굴을 성공적으로 훈련했습니다.")
print(f"모델이 '{trainer_file}' 파일로 저장되었습니다.")