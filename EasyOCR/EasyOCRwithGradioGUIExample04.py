import gradio as gr
import easyocr
import cv2
import numpy as np
from pdf2image import convert_from_bytes

# Initialize EasyOCR reader for Korean and English
reader = easyocr.Reader(['ko', 'en'])

def ocr_image_or_pdf(file):
    # If file is a numpy array, it's an image
    if isinstance(file, np.ndarray):
        image_bgr = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
        results = reader.readtext(image_bgr)
        output = [f"{text} (Confidence: {conf:.2f})" for _, text, conf in results]

        image_boxed = image_bgr.copy()
        for bbox, text, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(image_boxed, [pts], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(image_boxed, text, (pts[0][0], pts[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
        image_boxed_rgb = cv2.cvtColor(image_boxed, cv2.COLOR_BGR2RGB)
        return image_boxed_rgb, "\n".join(output) if output else "No text detected."
    
    # If file is bytes, it's a PDF
    elif isinstance(file, bytes):
        pages = convert_from_bytes(file)
        all_text = []
        images_with_boxes = []
        for i, page in enumerate(pages):
            page_np = np.array(page)
            image_bgr = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
            results = reader.readtext(image_bgr)
            page_output = [f"{text} (Confidence: {conf:.2f})" for _, text, conf in results]
            if page_output:
                all_text.append(f"--- Page {i+1} ---")
                all_text.extend(page_output)

            image_boxed = image_bgr.copy()
            for bbox, text, conf in results:
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(image_boxed, [pts], isClosed=True, color=(0,255,0), thickness=2)
                cv2.putText(image_boxed, text, (pts[0][0], pts[0][1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
            image_boxed_rgb = cv2.cvtColor(image_boxed, cv2.COLOR_BGR2RGB)
            images_with_boxes.append(image_boxed_rgb)
        # For PDFs, show the first page with boxes and all text
        if images_with_boxes:
            return images_with_boxes[0], "\n".join(all_text) if all_text else "No text detected."
        else:
            return None, "No text detected."
    else:
        return None, "Unsupported file type."

iface = gr.Interface(
    fn=ocr_image_or_pdf,
    inputs=gr.File(label="Upload Image (JPG, PNG, etc.) or PDF", type="filepath"),
    outputs=[
        gr.Image(type="numpy", label="Image with Recognized Boxes (First Page for PDF)"),
        gr.Textbox(label="Recognized Text"),
    ],
    title="Korean & English OCR (EasyOCR + Gradio)",
    description="Upload an image file (JPG, PNG, etc.) or PDF containing Korean or English text. The processed image with boxes will be shown first, followed by the recognized text.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()