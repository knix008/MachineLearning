import gradio as gr
import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize EasyOCR reader for Korean and English
reader = easyocr.Reader(['ko', 'en'])

def ocr_image_with_box(image):
    # Convert RGB (Gradio) to BGR (OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = reader.readtext(image_bgr)
    output = [f"{text} (Confidence: {conf:.2f})" for _, text, conf in results]

    # Draw boxes and labels
    image_boxed = image_bgr.copy()
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_boxed, [pts], isClosed=True, color=(0,255,0), thickness=2)
        #cv2.putText(image_boxed, text, (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
    # Convert back to RGB for display
    image_boxed_rgb = cv2.cvtColor(image_boxed, cv2.COLOR_BGR2RGB)
    return "\n".join(output) if output else "No text detected.", image_boxed_rgb

iface = gr.Interface(
    fn=ocr_image_with_box,
    inputs=gr.Image(type="numpy", label="Upload Image (JPG, PNG, etc.)"),
    outputs=[
        gr.Textbox(label="Recognized Text"),
        gr.Image(type="numpy", label="Image with Recognized Boxes"),
    ],
    title="Korean & English OCR (EasyOCR + Gradio)",
    description="Upload an image file (JPG, PNG, etc.) containing Korean or English text. The recognized text will be displayed below, and a separate window will show detected regions with boxes.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()