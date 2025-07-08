import cv2
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Video with OpenCV Modes")

        # Video Capture
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the camera.")
            root.destroy()
            return

        # 해상도 실제 값 얻기
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.mode = tk.StringVar(value="normal")

        # FPS calculation
        self.prev_time = time.time()
        self.fps_ema = 0

        # Video Writer 준비 (녹화 버튼 누를 때 시작)
        self.save_fps = 15.0
        self.out = None
        self.is_recording = False

        # Tkinter UI
        self.video_label = tk.Label(root)
        self.video_label.pack()

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=8)
        ttk.Button(
            button_frame, text="Normal", command=lambda: self.set_mode("normal")
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            button_frame, text="Threshold", command=lambda: self.set_mode("threshold")
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            button_frame, text="Edge", command=lambda: self.set_mode("edge")
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            button_frame, text="BG Sub", command=lambda: self.set_mode("bg_sub")
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            button_frame, text="Contour", command=lambda: self.set_mode("contour")
        ).pack(side=tk.LEFT, padx=2)
        self.record_btn = ttk.Button(
            button_frame, text="Start Recording", command=self.toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Quit", command=self.quit_app).pack(
            side=tk.LEFT, padx=2
        )

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.update_frame()

    def set_mode(self, mode):
        self.mode.set(mode)

    def toggle_recording(self):
        if not self.is_recording:
            # Works fine #
            #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            #self.out = cv2.VideoWriter(
            #    "output.mp4", fourcc, self.save_fps, (self.width, self.height)
            #)
            
            # XVID Warning #
            #fourcc = cv2.VideoWriter_fourcc(*"XVID")
            #self.out = cv2.VideoWriter(
            #    "output.mp4", fourcc, self.save_fps, (self.width, self.height)
            #)
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self.out = cv2.VideoWriter(
                "output.mp4", fourcc, self.save_fps, (self.width, self.height)
            )
            if not self.out.isOpened():
                messagebox.showerror(
                    "Recording Error", "Failed to initialize video writer!"
                )
                self.out = None
                return
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
        else:
            self.is_recording = False
            if self.out:
                self.out.release()
                self.out = None
            self.record_btn.config(text="Start Recording")

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            self.root.after(100, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        mode = self.mode.get()
        if mode == "threshold":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, display_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        elif mode == "edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == "bg_sub":
            fg_mask = self.bg_subtractor.apply(frame)
            display_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
        elif mode == "contour":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            display_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

        # Calculate FPS (EMA for smoothness)
        curr_time = time.time()
        inst_fps = 1 / (curr_time - self.prev_time)
        self.fps_ema = (
            (self.fps_ema * 0.9 + inst_fps * 0.1) if self.fps_ema else inst_fps
        )
        self.prev_time = curr_time

        # Overlay FPS and mode
        cv2.putText(
            display_frame,
            f"FPS: {int(self.fps_ema)} Mode: {mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Write frame to video if recording
        if self.is_recording and self.out:
            # frame 크기와 VideoWriter 크기가 같은지 확인
            if (display_frame.shape[1], display_frame.shape[0]) == (
                self.width,
                self.height,
            ):
                self.out.write(display_frame)
            else:
                resized = cv2.resize(display_frame, (self.width, self.height))
                self.out.write(resized)

        # Convert to Tkinter image
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def quit_app(self):
        if hasattr(self, "capture") and self.capture.isOpened():
            self.capture.release()
        if hasattr(self, "out") and self.out:
            self.out.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
