import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from huggingface_hub import hf_hub_download


class FaceDetectionApp:
    def __init__(self, window, window_title, model_path):
        self.window = window
        self.window.title(window_title)
        self.model = YOLO(model_path)

        self.video_source = 0
        self.vid = None
        self.running = False

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        self.btn_open = tk.Button(
            window, text="영상 파일 열기", width=20, command=self.open_file
        )
        self.btn_open.pack(anchor=tk.CENTER, expand=True)

        self.btn_cam = tk.Button(
            window, text="웹캠 실행", width=20, command=self.open_cam
        )
        self.btn_cam.pack(anchor=tk.CENTER, expand=True)

        self.btn_stop = tk.Button(window, text="정지", width=20, command=self.stop)
        self.btn_stop.pack(anchor=tk.CENTER, expand=True)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_source = file_path
            self.start_video()

    def open_cam(self):
        self.video_source = 0
        self.start_video()

    def start_video(self):
        self.stop()
        self.vid = cv2.VideoCapture(self.video_source)
        self.running = True
        self.update()

    def stop(self):
        self.running = False
        if self.vid is not None:
            self.vid.release()
            self.vid = None

    def update(self):
        if self.running and self.vid is not None:
            ret, frame = self.vid.read()
            if ret:
                results = self.model.predict(frame, conf=0.5)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = f"{self.model.names[cls]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                # 이미지 변환 및 표시
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img.resize((640, 480)))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.window.imgtk = imgtk  # 참조 유지

            self.window.after(10, self.update)
        else:
            if self.vid is not None:
                self.vid.release()
                self.vid = None

    def on_closing(self):
        self.stop()
        self.window.destroy()


def main():
    # Hugging Face에서 YOLOv8 얼굴 인식 모델 다운로드
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt"
    )
    FaceDetectionApp(tk.Tk(), "YOLOv8 얼굴 인식 (Tkinter)", model_path)


if __name__ == "__main__":
    model = "yolov8n-face.pt"  # YOLOv8 얼굴 인식 모델 경로
    main()
