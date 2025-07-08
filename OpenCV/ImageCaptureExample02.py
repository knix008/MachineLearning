import cv2
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Video with OpenCV Modes")

        # Video Capture
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.mode = tk.StringVar(value="normal")

        # FPS calculation
        self.prev_time = time.time()

        # Video Writer
        self.save_fps = 15.0
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.out = cv2.VideoWriter("output.avi", fourcc, self.save_fps, (1024, 576))

        # Tkinter UI
        self.video_label = tk.Label(root)
        self.video_label.pack()

        button_frame = ttk.Frame(root)
        button_frame.pack(pady=8)
        ttk.Button(button_frame, text="Normal", command=lambda: self.set_mode("normal")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Threshold", command=lambda: self.set_mode("threshold")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Edge", command=lambda: self.set_mode("edge")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="BG Sub", command=lambda: self.set_mode("bg_sub")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Contour", command=lambda: self.set_mode("contour")).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Quit", command=self.quit_app).pack(side=tk.LEFT, padx=2)

        self.update_frame()

    def set_mode(self, mode):
        self.mode.set(mode)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            self.root.after(30, self.update_frame)
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
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            display_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

        # Calculate FPS
        curr_time = time.time()
        processing_fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        # Overlay FPS and mode
        cv2.putText(
            display_frame,
            f"FPS: {int(processing_fps)} Mode: {mode}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Write frame to video
        self.out.write(display_frame)

        # Convert to Tkinter image
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(1, self.update_frame)

    def quit_app(self):
        self.capture.release()
        self.out.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()