import gradio as gr
import easyocr
import cv2
import numpy as np

# Initialize EasyOCR reader for Korean and English
reader = easyocr.Reader(['ko', 'en'])

def ocr_image(image):
    # Gradio gives the image as a numpy array in RGB
    # Convert to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = reader.readtext(image_bgr)
    output = [f"{text} (Confidence: {conf:.2f})" for _, text, conf in results]
    return "\n".join(output) if output else "No text detected."

iface = gr.Interface(
    fn=ocr_image,
    inputs=gr.Image(type="numpy", label="Upload Image (JPG, PNG, etc.)"),
    outputs=gr.Textbox(label="Recognized Text"),
    title="Korean & English OCR (EasyOCR + Gradio)",
    description="Upload an image file (JPG, PNG, etc.) containing Korean or English text. The recognized text will be displayed below.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()