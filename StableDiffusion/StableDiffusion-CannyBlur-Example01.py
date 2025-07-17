import gradio as gr
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import cv2
import numpy as np
from PIL import Image

def gaussian_blur_processing(img, kernel_size=50):
    """가우시안 블러 처리 - 원본 크기 및 비율 유지"""
    if img is None:
        return None
    
    # 홀수 커널 크기로 조정
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 원본 크기 저장
    original_size = img.size
    
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size)
    blurred_image = gaussian_blur(img)
    
    # 크기가 변경되었다면 원본 크기로 복원 (일반적으로 블러는 크기를 변경하지 않음)
    if blurred_image.size != original_size:
        blurred_image = blurred_image.resize(original_size, Image.LANCZOS)
    
    return blurred_image

def canny_edge_detection(img, low_threshold=100, high_threshold=200):
    """Canny 엣지 검출 - 원본 크기 및 비율 유지"""
    if img is None:
        return None
    
    # 원본 크기 저장
    original_size = img.size
    original_mode = img.mode
    
    # PIL을 numpy로 변환 (비율 유지)
    img_np = np.array(img)
    
    # RGB to Grayscale (원본이 이미 그레이스케일이 아닌 경우)
    if len(img_np.shape) == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_np
    
    # Canny edge detection
    edges = cv2.Canny(img_gray, low_threshold, high_threshold)
    
    # numpy를 PIL로 변환 (원본 크기 유지)
    edges_pil = Image.fromarray(edges, mode='L')
    
    # 크기 확인 및 복원 (만약 필요하다면)
    if edges_pil.size != original_size:
        edges_pil = edges_pil.resize(original_size, Image.LANCZOS)
    
    return edges_pil

def process_all(img, blur_kernel_size, canny_low, canny_high):
    """모든 처리를 한번에 수행 - 원본 크기 및 비율 유지"""
    if img is None:
        return None, None
    
    # 원본 정보 출력 (디버깅용)
    print(f"원본 이미지 크기: {img.size}, 모드: {img.mode}")
    
    blur_result = gaussian_blur_processing(img, blur_kernel_size)
    canny_result = canny_edge_detection(img, canny_low, canny_high)
    
    # 결과 정보 출력 (디버깅용)
    if blur_result:
        print(f"블러 결과 크기: {blur_result.size}")
    if canny_result:
        print(f"Canny 결과 크기: {canny_result.size}")
    
    return blur_result, canny_result

def validate_image_format(img):
    """이미지 형식 검증 (선택적)"""
    if img is None:
        return False
    
    # PIL Image 객체에서는 format 속성으로 확인 가능
    allowed_formats = ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']
    return img.format in allowed_formats if hasattr(img, 'format') else True

def update_image_info(img):
    """이미지 정보 업데이트"""
    if img is None:
        return "이미지가 없습니다."
    
    # 형식 검증 (선택적)
    if not validate_image_format(img):
        return "지원되지 않는 이미지 형식입니다."
    
    return f"크기: {img.size[0]} x {img.size[1]}, 모드: {img.mode}"

# Gradio 인터페이스 생성
with gr.Blocks(title="이미지 처리 도구") as demo:
    gr.Markdown("# 이미지 처리 도구")
    gr.Markdown("이미지를 업로드하고 다양한 처리를 적용해보세요. (원본 크기 및 비율 유지)")
    gr.Markdown("**지원 형식**: JPG, JPEG, PNG, BMP, TIFF, WEBP")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", 
                label="입력 이미지",
                sources=["upload", "clipboard"]
            )
            
            # 이미지 정보 표시
            image_info = gr.Textbox(
                label="이미지 정보",
                interactive=False,
                placeholder="이미지를 업로드하면 크기 정보가 표시됩니다."
            )
            
            # 컨트롤 패널
            with gr.Group():
                blur_kernel = gr.Slider(
                    minimum=1, 
                    maximum=100, 
                    value=50, 
                    step=2, 
                    label="블러 커널 크기"
                )
                
                with gr.Row():
                    canny_low = gr.Slider(
                        minimum=50, 
                        maximum=150, 
                        value=100, 
                        step=10, 
                        label="Canny 낮은 임계값"
                    )
                    canny_high = gr.Slider(
                        minimum=150, 
                        maximum=300, 
                        value=200, 
                        step=10, 
                        label="Canny 높은 임계값"
                    )
            
            # 버튼들
            with gr.Group():
                process_btn = gr.Button("모든 처리 실행", variant="primary", size="lg")
                
                with gr.Row():
                    blur_btn = gr.Button("가우시안 블러")
                    canny_btn = gr.Button("엣지 검출")
        
        with gr.Column():
            blur_output = gr.Image(
                label="블러 처리 결과"
            )
            canny_output = gr.Image(
                label="엣지 검출 결과"
            )

    # 이벤트 핸들러 (gr.Blocks 컨텍스트 안에서 정의)
    input_image.change(
        fn=update_image_info,
        inputs=[input_image],
        outputs=[image_info]
    )

    process_btn.click(
        fn=process_all,
        inputs=[input_image, blur_kernel, canny_low, canny_high],
        outputs=[blur_output, canny_output]
    )

    blur_btn.click(
        fn=gaussian_blur_processing,
        inputs=[input_image, blur_kernel],
        outputs=[blur_output]
    )

    canny_btn.click(
        fn=canny_edge_detection,
        inputs=[input_image, canny_low, canny_high],
        outputs=[canny_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)