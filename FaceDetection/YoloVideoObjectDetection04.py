import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import tempfile
import os
import random

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # replace with your model if different


def get_color(idx):
    random.seed(idx)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


class VideoObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLOv8 객체 탐지 실시간 데모 (tkinter)")
        self.video_path = None
        self.cap = None

        # UI 구성
        self.btn_open = tk.Button(master, text="동영상 열기", command=self.open_video)
        self.btn_open.pack(pady=10)

        self.canvas = tk.Label(master)
        self.canvas.pack()

        self.status = tk.Label(master, text="", anchor="w")
        self.status.pack(fill="x")

        self.frame = None

    def open_video(self):
        video_path = filedialog.askopenfilename(
            title="객체 탐지할 영상을 선택하세요",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")],
        )
        if video_path:
            self.video_path = video_path
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            self.status.config(text=f"분석 중: {os.path.basename(self.video_path)}")
            self.show_frame()

    def show_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.status.config(text="동영상을 열 수 없습니다.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status.config(text="분석 완료 또는 프레임 없음.")
            self.cap.release()
            return

        # Object detection
        results = model(frame)
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls[i])
                label = (
                    model.model.names[class_id]
                    if hasattr(model.model, "names")
                    else str(class_id)
                )
                confidence = conf[i]
                color = get_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        # Convert BGR to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        self.canvas.img_tk = img_tk  # Prevent garbage collection
        self.canvas.config(image=img_tk)

        # 30ms 후 다음 프레임 갱신 (약 33fps)
        self.master.after(30, self.show_frame)


def main():
    root = tk.Tk()
    app = VideoObjectDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
