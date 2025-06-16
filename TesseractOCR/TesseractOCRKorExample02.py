import cv2
import numpy as np
import pytesseract
import gradio as gr
from PIL import Image

# Make sure Tesseract is installed and path is set if needed. Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_korean(image):
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run pytesseract with Korean language
    data = pytesseract.image_to_data(image_cv, lang='kor', output_type=pytesseract.Output.DICT)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i]
        if text.strip() != "":
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # Draw rectangle around detected Korean text
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Convert back to PIL Image for Gradio display
    result_img = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_img)

demo = gr.Interface(
    fn=recognize_korean,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Detected Korean Text"),
    title="Korean Text Recognition",
    description="Upload an image with Korean text. The app will recognize Korean text and display rectangles around detected regions."
)

if __name__ == "__main__":
    demo.launch()