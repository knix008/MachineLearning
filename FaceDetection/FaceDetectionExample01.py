import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch


def get_face(image_path):
    # Load image and convert to RGB
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use OpenCV's Haar cascade to detect face
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(rgb_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"No face found in {image_path}")
        return None

    # Crop the first face found
    x, y, w, h = faces[0]
    face = rgb_img[y : y + h, x : x + w]
    face_pil = Image.fromarray(face)
    return face_pil


def get_embedding(face_img, model):
    # Preprocess for facenet-pytorch
    face_img = face_img.resize((160, 160))
    face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
    face_tensor = (face_tensor - 0.5) / 0.5  # Normalize
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding[0]


def cosine_similarity(a, b):
    a = a / a.norm()
    b = b / b.norm()
    return torch.dot(a, b).item()


def is_same_face(img1_path, img2_path, threshold=0.6):
    model = InceptionResnetV1(pretrained="vggface2").eval()
    face1 = get_face(img1_path)
    face2 = get_face(img2_path)

    if face1 is None or face2 is None:
        return False

    emb1 = get_embedding(face1, model)
    emb2 = get_embedding(face2, model)

    sim = cosine_similarity(emb1, emb2)
    print(f"Cosine similarity: {sim:.3f}")
    return sim > threshold


if __name__ == "__main__":
    img1 = "person1.jpg"
    img2 = "person1.jpg"
    result = is_same_face(img1, img2)
    print("Faces are the same!" if result else "Faces are NOT the same!")
