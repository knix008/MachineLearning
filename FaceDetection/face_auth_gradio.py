import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import shutil

# OpenCV 얼굴 인식 모듈 import
try:
    from cv2 import face
except ImportError:
    print("OpenCV 얼굴 인식 모듈이 설치되지 않았습니다. 설치 중...")
    # 대안: 기본 얼굴 감지만 사용

class FaceAuthSystem:
    def __init__(self, registered_dir="Registered"):
        self.registered_dir = registered_dir
        self.ensure_registered_dir()
        
        # OpenCV 얼굴 감지기 초기화
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def ensure_registered_dir(self):
        """등록된 얼굴 이미지를 저장할 디렉토리가 없으면 생성"""
        if not os.path.exists(self.registered_dir):
            os.makedirs(self.registered_dir)
    
    def detect_face(self, img_array):
        """이미지에서 얼굴 감지"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces, gray
    
    def extract_face_region(self, img_array, face_coords):
        """얼굴 영역 추출"""
        x, y, w, h = face_coords
        face_roi = img_array[y:y+h, x:x+w]
        return face_roi
    
    def resize_face(self, face_img, size=(100, 100)):
        """얼굴 이미지 크기 조정"""
        return cv2.resize(face_img, size)
    
    def calculate_similarity(self, img1, img2):
        """두 이미지 간의 유사도 계산 (MSE 기반)"""
        # 이미지를 1차원 벡터로 변환
        img1_flat = img1.flatten().astype(float)
        img2_flat = img2.flatten().astype(float)
        
        # Mean Squared Error 계산
        mse = np.mean((img1_flat - img2_flat) ** 2)
        
        # MSE를 유사도로 변환 (낮은 MSE = 높은 유사도)
        # 최대 MSE를 255^2로 가정하고 정규화
        max_mse = 255 * 255
        similarity = max(0, 1 - (mse / max_mse))
        
        return similarity
    
    def register_face(self, image, name):
        """새로운 얼굴을 등록"""
        if image is None or name.strip() == "":
            return "이미지와 이름을 모두 입력해주세요."
        
        try:
            # PIL Image를 numpy array로 변환
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                # RGB to BGR (OpenCV format)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_array = image
            
            # 얼굴 감지
            faces, gray = self.detect_face(img_array)
            
            if len(faces) == 0:
                return "이미지에서 얼굴을 찾을 수 없습니다. 다른 이미지를 사용해주세요."
            
            if len(faces) > 1:
                return "이미지에 여러 개의 얼굴이 있습니다. 한 명의 얼굴만 포함된 이미지를 사용해주세요."
            
            # 얼굴 영역 추출 및 크기 조정
            face_roi = self.extract_face_region(gray, faces[0])
            face_resized = self.resize_face(face_roi)
            
            # 이름으로 파일 저장
            filename = f"{name.strip()}.jpg"
            filepath = os.path.join(self.registered_dir, filename)
            
            # 얼굴 영역을 저장
            cv2.imwrite(filepath, face_resized)
            
            return f"'{name}'의 얼굴이 성공적으로 등록되었습니다!"
            
        except Exception as e:
            return f"등록 중 오류가 발생했습니다: {str(e)}"
    
    def load_registered_faces(self):
        """등록된 얼굴들 로드"""
        registered_faces = []
        registered_names = []
        
        for filename in os.listdir(self.registered_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(self.registered_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    registered_faces.append(img)
                    registered_names.append(filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', ''))
        
        return registered_faces, registered_names
    
    def verify_face(self, image):
        """등록된 얼굴과 비교하여 인증"""
        if image is None:
            return "인증할 이미지를 업로드해주세요."
        
        try:
            # PIL Image를 numpy array로 변환
            if isinstance(image, Image.Image):
                img_array = np.array(image)
                # RGB to BGR (OpenCV format)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_array = image
            
            # 입력 이미지에서 얼굴 감지
            faces, gray = self.detect_face(img_array)
            
            if len(faces) == 0:
                return "이미지에서 얼굴을 찾을 수 없습니다."
            
            if len(faces) > 1:
                return "이미지에 여러 개의 얼굴이 있습니다. 한 명의 얼굴만 포함된 이미지를 사용해주세요."
            
            # 입력 이미지의 얼굴 영역 추출 및 크기 조정
            input_face_roi = self.extract_face_region(gray, faces[0])
            input_face_resized = self.resize_face(input_face_roi)
            
            # 등록된 얼굴들 로드
            registered_faces, registered_names = self.load_registered_faces()
            
            if not registered_faces:
                return "등록된 얼굴이 없습니다. 먼저 얼굴을 등록해주세요."
            
            # 등록된 얼굴들과 유사도 계산
            similarities = []
            for registered_face in registered_faces:
                similarity = self.calculate_similarity(input_face_resized, registered_face)
                similarities.append(similarity)
            
            # 가장 높은 유사도를 가진 얼굴 찾기
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            # 임계값 설정 (0.6 이상이면 인증 성공)
            threshold = 0.6
            
            if best_similarity >= threshold:
                name = registered_names[best_match_index]
                return f"✅ 인증 성공!\n\n인증된 사용자: {name}\n유사도: {best_similarity:.2%}"
            else:
                return f"❌ 인증 실패!\n\n등록되지 않은 사용자입니다.\n최고 유사도: {best_similarity:.2%}"
                
        except Exception as e:
            return f"인증 중 오류가 발생했습니다: {str(e)}"
    
    def get_registered_faces(self):
        """등록된 얼굴 목록 반환"""
        faces = []
        for filename in os.listdir(self.registered_dir):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                name = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
                faces.append(name)
        return faces

# 얼굴 인증 시스템 인스턴스 생성
face_system = FaceAuthSystem()

def register_face_interface(image, name):
    """얼굴 등록 인터페이스"""
    return face_system.register_face(image, name)

def verify_face_interface(image):
    """얼굴 인증 인터페이스"""
    return face_system.verify_face(image)

def get_registered_list():
    """등록된 얼굴 목록 반환"""
    faces = face_system.get_registered_faces()
    if faces:
        return "등록된 사용자:\n" + "\n".join([f"• {face}" for face in faces])
    else:
        return "등록된 사용자가 없습니다."

# Gradio 인터페이스 생성
with gr.Blocks(title="얼굴 인증 시스템") as demo:
    gr.Markdown("# 👤 얼굴 인증 시스템")
    gr.Markdown("사람의 얼굴을 등록하고 인식하여 인증하는 시스템입니다.")
    
    with gr.Tab("얼굴 등록"):
        gr.Markdown("### 새로운 사용자 등록")
        gr.Markdown("등록할 사람의 얼굴이 명확히 보이는 이미지를 업로드하고 이름을 입력하세요.")
        
        with gr.Row():
            with gr.Column():
                register_image = gr.Image(label="등록할 얼굴 이미지", type="pil")
                register_name = gr.Textbox(label="이름", placeholder="등록할 사람의 이름을 입력하세요")
                register_btn = gr.Button("얼굴 등록", variant="primary")
            
            with gr.Column():
                register_result = gr.Textbox(label="등록 결과", lines=3, interactive=False)
                registered_list = gr.Textbox(label="등록된 사용자 목록", lines=5, interactive=False)
        
        register_btn.click(
            fn=register_face_interface,
            inputs=[register_image, register_name],
            outputs=register_result
        ).then(
            fn=get_registered_list,
            outputs=registered_list
        )
    
    with gr.Tab("얼굴 인증"):
        gr.Markdown("### 사용자 인증")
        gr.Markdown("인증할 사람의 얼굴이 명확히 보이는 이미지를 업로드하세요.")
        
        with gr.Row():
            with gr.Column():
                verify_image = gr.Image(label="인증할 얼굴 이미지", type="pil")
                verify_btn = gr.Button("얼굴 인증", variant="primary")
            
            with gr.Column():
                verify_result = gr.Textbox(label="인증 결과", lines=5, interactive=False)
        
        verify_btn.click(
            fn=verify_face_interface,
            inputs=verify_image,
            outputs=verify_result
        )
    
    with gr.Tab("등록된 사용자"):
        gr.Markdown("### 등록된 사용자 목록")
        refresh_btn = gr.Button("목록 새로고침")
        users_list = gr.Textbox(label="등록된 사용자", lines=10, interactive=False)
        
        refresh_btn.click(
            fn=get_registered_list,
            outputs=users_list
        )
        
        # 초기 로드
        demo.load(
            fn=get_registered_list,
            outputs=users_list
        )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860) 