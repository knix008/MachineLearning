# 필요한 라이브러리를 설치해야 합니다.
# 터미널에서 다음 명령어를 실행하세요:
# pip install opencv-python
# pip install pillow

import cv2
import os
import requests
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# 얼굴 인식을 위한 Haar Cascade 분류기 XML 파일을 다운로드합니다.
# 이 파일은 얼굴 인식 알고리즘에 필수적입니다.
# 파일이 존재하는지 확인하고, 없으면 다운로드를 시도합니다.
xml_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_path):
    print("Haar Cascade 파일을 다운로드 중입니다...")
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        r = requests.get(url, allow_redirects=True)
        with open(xml_path, 'wb') as f:
            f.write(r.content)
        print("다운로드 완료.")
    except Exception as e:
        print(f"Haar Cascade 파일 다운로드 오류: {e}")
        print(f"수동으로 '{url}'에서 파일을 다운로드하여 이 스크립트와 동일한 디렉토리에 저장해주세요.")
        exit()

# 미리 학습된 Haar Cascade 모델을 로드합니다.
try:
    face_cascade = cv2.CascadeClassifier(xml_path)
    if face_cascade.empty():
        raise IOError("Haar Cascade 분류기 로드 실패.")
except Exception as e:
    print(f"오류: {e}")
    print(f"'{xml_path}' 파일이 올바른 디렉토리에 있고 손상되지 않았는지 확인해주세요.")
    exit()

class FaceDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # 웹캠 초기화
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("웹캠을 열 수 없습니다.")

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # GUI 요소 생성
        self.create_widgets()
        
        # 비디오 루프 시작을 위해 첫 프레임 업데이트 예약
        self.delay = 15  # 프레임 업데이트 간격 (ms)
        self.update_frame()

        # 창 닫기 이벤트 처리
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 전체 프레임
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목 레이블
        title_label = ttk.Label(main_frame, text="실시간 얼굴 인식 (입력 vs 출력)", font=("Helvetica", 16))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # 입력(원본) 비디오 패널
        input_frame = ttk.LabelFrame(main_frame, text="입력 (원본 영상)")
        input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.input_panel = ttk.Label(input_frame)
        self.input_panel.pack(padx=5, pady=5)
        
        # 출력(결과) 비디오 패널
        output_frame = ttk.LabelFrame(main_frame, text="출력 (얼굴 인식)")
        output_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.output_panel = ttk.Label(output_frame)
        self.output_panel.pack(padx=5, pady=5)

    def update_frame(self):
        """웹캠에서 프레임을 읽고 GUI를 업데이트합니다."""
        ret, frame = self.vid.read()
        
        if ret:
            # OpenCV는 BGR 형식이지만 Pillow는 RGB 형식을 사용하므로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 얼굴 인식 수행
            detected_frame = self.detect_faces(frame.copy())
            detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

            # Tkinter에서 표시할 수 있도록 이미지 형식 변환
            # 이미지가 가비지 컬렉션되는 것을 방지하기 위해 인스턴스 변수로 저장해야 합니다.
            self.original_photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.detected_photo = ImageTk.PhotoImage(image=Image.fromarray(detected_frame_rgb))

            # GUI 패널 업데이트
            self.input_panel.config(image=self.original_photo)
            self.output_panel.config(image=self.detected_photo)
            
            # 다음 프레임 업데이트를 예약
            self.window.after(self.delay, self.update_frame)
        else:
            print("프레임을 읽을 수 없습니다. 스트림 종료.")
            self.on_closing()

    def detect_faces(self, image):
        """입력 이미지에서 얼굴을 감지하고 사각형을 그립니다."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        for (x, y, w, h) in faces:
            # 얼굴 주위에 파란색 사각형 그리기
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        return image

    def on_closing(self):
        """애플리케이션 종료 시 호출됩니다."""
        print("애플리케이션 종료 중...")
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    # Tkinter 창 생성 및 애플리케이션 시작
    root = tk.Tk()
    app = FaceDetectorApp(root, "Tkinter 얼굴 인식 프로그램")
    root.mainloop()
