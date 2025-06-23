import cv2
import numpy as np
import os

def load_registered_images(registered_dir):
    registered_faces = []
    registered_names = []

    for filename in os.listdir(registered_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(registered_dir, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            registered_faces.append(gray)
            registered_names.append(filename)
    return registered_faces, registered_names

def verify_face(input_image_path, registered_dir, threshold=0.6):
    # Load input image and convert to grayscale
    input_img = cv2.imread(input_image_path)
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Load registered faces
    registered_faces, registered_names = load_registered_images(registered_dir)

    # Create face recognizer (LBPH for simplicity)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train with registered images (labels are just indices)
    face_recognizer.train(registered_faces, np.array(range(len(registered_faces))))

    # Recognize face in input image
    label, confidence = face_recognizer.predict(input_gray)
    print(f"Best match: {registered_names[label]}, Confidence: {confidence}")

    # LBPH confidence: smaller is better, so we invert the threshold logic
    if confidence < threshold * 100:  # LBPH confidence is typically 0~100
        print(f"입력 이미지가 등록된 얼굴과 일치합니다: {registered_names[label]}", "Label : ", label)
        return True, registered_names[label]
    else:
        print("입력 이미지가 등록된 얼굴과 일치하지 않습니다.")
        return False, None

if __name__ == "__main__":
    # 예시 사용법
    registered_dir = "Registered"         # 등록된 이미지 폴더
    input_image_path = "person3.jpg"      # 비교할 입력 이미지
    verify_face(input_image_path, registered_dir)