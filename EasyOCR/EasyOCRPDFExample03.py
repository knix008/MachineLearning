import easyocr
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
import fitz

import torch
import time
import datetime
import warnings

# Suppress warnings from EasyOCR
warnings.filterwarnings("ignore", category=UserWarning)

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print("> Using device:", device)

def draw_boxes(image, results):
    """Draw bounding boxes for recognized text."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for (bbox, text, conf) in results:
        points = [tuple(point) for point in bbox]
        draw.polygon(points, outline='red', width=2)
    return image

def process_file(input_file):
    if input_file is None:
        return None, None, ""
    # Initialize EasyOCR reader for Korean and English
    reader = easyocr.Reader(['ko', 'en'], gpu=True if device == "cuda" else False)
    # Handle PDF file
    if hasattr(input_file, "name") and input_file.name.lower().endswith('.pdf'):
        pages = fitz.open(input_file.name)
        if not pages:
            return None, None, "No pages found in PDF."
        image = pages[0].get_pixmap()   # Get the first page as an image    
        print("The Covverted image size:", image.width, image.height)
        image = Image.frombytes("RGB", [image.width, image.height], image.samples)
        # Convert to numpy array for EasyOCR
        np_img = np.array(image)
        input_display_image = image
        start = time.time()
        results = reader.readtext(np_img)
        end = time.time()
        print(f">> Processing time for PDF page: {datetime.timedelta(seconds=end - start)}")
    else:
        image = Image.open(input_file).convert("RGB")
        print("Image size:", image.width, image.height)
        np_img = np.array(image)
        input_display_image = image
        start = time.time()
        results = reader.readtext(np_img)
        end = time.time()
        print(f">> Processing time for image: {datetime.timedelta(seconds=end - start)}")

    boxed_img = draw_boxes(image, results)
    texts = [f"{text} (conf: {conf:.2f})" for _, text, conf in results]
    all_text = "\n".join(texts)
    return input_display_image, boxed_img, all_text

with gr.Blocks() as demo:
    gr.Markdown("# Korean & English OCR (EasyOCR) - PDF and Image")
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Upload Image or PDF", file_types=["image", ".pdf"])
            input_display_display = gr.Image(label="Input Image/PDF First Page")
            run_btn = gr.Button("Run OCR")
        with gr.Column():
            output_image = gr.Image(label="Image with OCR Boxes")
            output_text = gr.Textbox(label="Recognized Text")
    run_btn.click(process_file, inputs=input_file, outputs=[input_display_display, output_image, output_text])

if __name__ == "__main__":
    demo.launch()