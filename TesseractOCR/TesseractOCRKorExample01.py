import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def ocr_image(image_path, lang='eng+kor'):
    """Perform OCR on a single image file, supporting English and Korean."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def ocr_pdf(pdf_path, lang='eng+kor'):
    """Perform OCR on a PDF file by converting each page to an image, supporting English and Korean."""
    pages = convert_from_path(pdf_path)
    full_text = ""
    for page_number, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page, lang=lang)
        full_text += f"\n\n--- Page {page_number} ---\n{text}"
    return full_text

if __name__ == '__main__':
    # Example usage
    image_file = 'test_kor.png'
    pdf_file = 'test_kor.png'
    
    print("OCR result for image file (English+Korean):")
    print(ocr_image(image_file, lang='eng+kor'))