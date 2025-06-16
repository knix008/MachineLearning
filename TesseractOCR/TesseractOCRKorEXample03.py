import cv2
import numpy as np
import pytesseract
import gradio as gr
from PIL import Image

# Make sure Tesseract is installed and path is set if needed.
# Example for Windows users:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_korean(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    data = pytesseract.image_to_data(image_cv, lang='kor', output_type=pytesseract.Output.DICT)
    recognized_text = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i]
        if text.strip() != "":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            recognized_text.append(text.strip())
    result_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Join the recognized characters line by line
    recognized_text_display = "\n".join(recognized_text) if recognized_text else "No Korean text detected."
    return Image.fromarray(result_img), recognized_text_display

demo = gr.Interface(
    fn=recognize_korean,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Detected Korean Text"),
        gr.Textbox(label="Recognized Characters", lines=6)
    ],
    title="Korean Text Recognition",
    description="Upload an image with Korean text. The app will recognize Korean text, display rectangles around detected regions, and show the recognized characters in a separate field."
)

if __name__ == "__main__":
    demo.launch()