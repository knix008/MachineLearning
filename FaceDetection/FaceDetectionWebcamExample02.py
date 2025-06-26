# First, you need to install the required libraries.
# You can do this by running the following commands in your terminal:
# pip install gradio
# pip install opencv-python

import cv2
import gradio as gr
import numpy as np
import os
import requests

# Download the Haar Cascade classifier XML file for face detection.
# This file is necessary for the face detection algorithm.
# We will check if the file exists, and if not, we'll try to download it.
# A more robust solution would be to include it with your project files.
xml_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_path):
    print("Downloading Haar Cascade file...")
    try:
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        r = requests.get(url, allow_redirects=True)
        with open(xml_path, 'wb') as f:
            f.write(r.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading the Haar Cascade file: {e}")
        print("Please download it manually from 'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml' and place it in the same directory as this script.")
        exit()

# Load the pre-trained Haar Cascade model for face detection.
# This model is effective for detecting frontal faces.
try:
    face_cascade = cv2.CascadeClassifier(xml_path)
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade classifier.")
except Exception as e:
    print(f"Error: {e}")
    print("Please ensure the 'haarcascade_frontalface_default.xml' file is in the correct directory and is not corrupted.")
    exit()

def detect_faces(image):
    """
    This function takes an image as input, detects faces in it,
    and returns the image with rectangles drawn around the detected faces.

    Args:
        image (numpy.ndarray): The input image from the webcam.

    Returns:
        numpy.ndarray: The image with detected faces highlighted.
                       Returns the original image if no faces are detected or if there is an error.
    """
    if image is None:
        # Return a black placeholder image if the input is None
        return np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        # Convert the image to grayscale. Face detection is performed on grayscale images.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image.
        # - scaleFactor: How much the image size is reduced at each image scale.
        # - minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        # - minSize: Minimum possible object size. Objects smaller than this are ignored.
        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            # The rectangle is drawn on the original color image.
            # (x, y) is the top-left corner of the rectangle.
            # (x+w, y+h) is the bottom-right corner.
            # (255, 0, 0) is the color of the rectangle (Blue in BGR).
            # 10 is the thickness of the rectangle border.
            # FIX: Explicitly cast coordinates to Python integers to prevent type errors.
            cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 10)

        return image

    except Exception as e:
        print(f"An error occurred during face detection: {e}")
        # Return the original image in case of an error
        return image

# Create the Gradio interface
# `streaming=True` enables real-time video processing from the webcam.
# The `live=True` parameter in the launch method also helps with continuous updates.
iface = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="Real-time Face Detection",
    description="This application uses your webcam to detect faces in real-time. "
                "Allow camera access when prompted by your browser."
)

if __name__ == "__main__":
    # Launch the Gradio web server.
    # You can access the interface at the local URL provided in the terminal.
    # Set share=True to create a public link (useful for sharing).
    iface.launch(share=False)
