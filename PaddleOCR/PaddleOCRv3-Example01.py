from paddleocr import PaddleOCR
 
ocr = PaddleOCR(lang="korean")
 
img_path = "assets/images/test_image_1.jpg"
result = ocr.ocr(img_path, cls=False)
 
ocr_result = result[0]
print(ocr_result)