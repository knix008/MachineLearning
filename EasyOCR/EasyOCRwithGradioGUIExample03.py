import gradio as gr
import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['ko', 'en'])

def ocr_image_with_box(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = reader.readtext(image_bgr)
    output = [f"{text} (Confidence: {conf:.2f})" for _, text, conf in results]

    image_boxed = image_bgr.copy()
    for bbox, text, conf in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_boxed, [pts], isClosed=True, color=(0,255,0), thickness=2)

    image_boxed_rgb = cv2.cvtColor(image_boxed, cv2.COLOR_BGR2RGB)
    # Return IMAGE FIRST, then TEXT
    return image_boxed_rgb, "\n".join(output) if output else "No text detected."

iface = gr.Interface(
    fn=ocr_image_with_box,
    inputs=gr.Image(type="numpy", label="Upload Image (JPG, PNG, etc.)"),
    outputs=[
        gr.Image(type="numpy", label="Image with Recognized Boxes"),
        gr.Textbox(label="Recognized Text"),
    ],
    title="Korean & English OCR (EasyOCR + Gradio)",
    description="Upload an image file (JPG, PNG, etc.) containing Korean or English text. The processed image with boxes will be shown first, followed by the recognized text.",
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch()