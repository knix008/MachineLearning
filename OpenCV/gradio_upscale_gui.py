import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

def upscale_image_4x(image, interpolation_method='cubic', enhance_edges=False):
    """
    이미지를 4배 확대하는 함수
    
    Args:
        image: PIL Image 또는 numpy array
        interpolation_method: 보간법 ('nearest', 'bilinear', 'cubic', 'lanczos')
        enhance_edges: 엣지 강화 적용 여부
    
    Returns:
        tuple: (원본 이미지, 확대된 이미지, 정보 텍스트)
    """
    
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image_np = np.array(image)
        # RGB를 BGR로 변환 (OpenCV용)
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_np = image
    
    # 원본 크기
    height, width = image_np.shape[:2]
    original_size = f"원본 크기: {width} x {height}"
    
    # 보간법 매핑
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interpolation = interpolation_map.get(interpolation_method, cv2.INTER_CUBIC)
    
    # 4배 확대
    upscaled = cv2.resize(image_np, None, fx=4.0, fy=4.0, interpolation=interpolation)
    
    # 엣지 강화 적용
    if enhance_edges:
        # 언샤프 마스킹으로 엣지 강화
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
        upscaled = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
        
        # 노이즈 제거
        upscaled = cv2.fastNlMeansDenoisingColored(upscaled, None, 10, 10, 7, 21)
    
    # BGR을 RGB로 변환 (PIL용)
    if len(upscaled.shape) == 3:
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    else:
        upscaled_rgb = upscaled
    
    # 확대된 크기
    new_height, new_width = upscaled_rgb.shape[:2]
    upscaled_size = f"확대된 크기: {new_width} x {new_height}"
    
    # 메모리 사용량 계산
    original_memory = width * height * 3 / 1024 / 1024  # MB
    upscaled_memory = new_width * new_height * 3 / 1024 / 1024  # MB
    
    info_text = f"""
{original_size}
{upscaled_size}
보간법: {interpolation_method.upper()}
엣지 강화: {'적용' if enhance_edges else '미적용'}
원본 메모리: {original_memory:.1f} MB
확대 메모리: {upscaled_memory:.1f} MB
메모리 증가: {upscaled_memory/original_memory:.1f}배
"""
    
    # 원본 이미지를 RGB로 변환
    if len(image_np.shape) == 3:
        original_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = image_np
    
    return original_rgb, upscaled_rgb, info_text

def compare_methods(image):
    """
    다양한 보간법으로 확대하여 비교하는 함수
    
    Args:
        image: PIL Image
    
    Returns:
        tuple: (원본, nearest, bilinear, cubic, lanczos)
    """
    
    if image is None:
        return None, None, None, None, None
    
    # PIL Image를 numpy array로 변환
    image_np = np.array(image)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 보간법들
    methods = [
        ('nearest', cv2.INTER_NEAREST),
        ('bilinear', cv2.INTER_LINEAR),
        ('cubic', cv2.INTER_CUBIC),
        ('lanczos', cv2.INTER_LANCZOS4)
    ]
    
    results = []
    
    for method_name, interpolation in methods:
        # 4배 확대
        upscaled = cv2.resize(image_np, None, fx=4.0, fy=4.0, interpolation=interpolation)
        
        # BGR을 RGB로 변환
        if len(upscaled.shape) == 3:
            upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        else:
            upscaled_rgb = upscaled
        
        results.append(upscaled_rgb)
    
    # 원본 이미지
    if len(image_np.shape) == 3:
        original_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = image_np
    
    return original_rgb, *results

def save_upscaled_image(image, filename="upscaled_4x.jpg"):
    """
    확대된 이미지를 저장하는 함수
    
    Args:
        image: numpy array (RGB)
        filename: 저장할 파일명
    
    Returns:
        str: 저장된 파일 경로
    """
    
    if image is None:
        return None
    
    # RGB를 BGR로 변환 (OpenCV 저장용)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 임시 파일로 저장
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    cv2.imwrite(file_path, image_bgr)
    return file_path

# Gradio 인터페이스 생성
def create_interface():
    with gr.Blocks(title="OpenCV 4배 이미지 확대", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 OpenCV 4배 이미지 확대 도구")
        gr.Markdown("이미지를 4배 확대하고 다양한 보간법을 비교해보세요.")
        
        with gr.Tab("기본 확대"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 입력 섹션
                    gr.Markdown("### 📤 이미지 업로드")
                    input_image = gr.Image(label="원본 이미지", type="pil")
                    
                    gr.Markdown("### ⚙️ 설정")
                    interpolation_method = gr.Dropdown(
                        choices=["cubic", "bilinear", "nearest", "lanczos"],
                        value="cubic",
                        label="보간법 선택",
                        info="cubic: 높은 품질 (권장), bilinear: 적당한 품질, nearest: 빠름, lanczos: 최고 품질"
                    )
                    
                    enhance_edges = gr.Checkbox(
                        label="엣지 강화 적용",
                        value=False,
                        info="언샤프 마스킹과 노이즈 제거를 적용합니다"
                    )
                    
                    upscale_btn = gr.Button("🚀 4배 확대", variant="primary")
                    
                with gr.Column(scale=2):
                    # 출력 섹션
                    gr.Markdown("### 📊 결과")
                    with gr.Row():
                        original_output = gr.Image(label="원본 이미지", height=300)
                        upscaled_output = gr.Image(label="4배 확대된 이미지", height=300)
                    
                    info_output = gr.Textbox(
                        label="📋 처리 정보",
                        lines=8,
                        interactive=False
                    )
                    
                    download_btn = gr.File(label="💾 다운로드")
        
        with gr.Tab("보간법 비교"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 이미지 업로드")
                    compare_input = gr.Image(label="원본 이미지", type="pil")
                    compare_btn = gr.Button("🔄 보간법 비교", variant="primary")
                
                with gr.Column(scale=3):
                    gr.Markdown("### 📊 보간법별 결과 비교")
                    with gr.Row():
                        compare_original = gr.Image(label="원본", height=200)
                        compare_nearest = gr.Image(label="Nearest Neighbor", height=200)
                        compare_bilinear = gr.Image(label="Bilinear", height=200)
                    with gr.Row():
                        compare_cubic = gr.Image(label="Bicubic", height=200)
                        compare_lanczos = gr.Image(label="Lanczos", height=200)
        
        with gr.Tab("정보"):
            gr.Markdown("""
            ## 📚 사용법
            
            ### 1. 기본 확대
            1. 이미지를 업로드합니다
            2. 보간법을 선택합니다
            3. 엣지 강화 옵션을 설정합니다
            4. "4배 확대" 버튼을 클릭합니다
            5. 결과를 확인하고 다운로드합니다
            
            ### 2. 보간법 비교
            1. 이미지를 업로드합니다
            2. "보간법 비교" 버튼을 클릭합니다
            3. 각 보간법의 결과를 비교합니다
            
            ## 🔧 보간법 설명
            
            - **Nearest Neighbor**: 가장 빠르지만 품질이 낮음
            - **Bilinear**: 적당한 속도와 품질
            - **Bicubic**: 높은 품질 (권장)
            - **Lanczos**: 최고 품질, 처리 시간이 오래 걸림
            
            ## ⚠️ 주의사항
            
            - 4배 확대시 메모리 사용량이 16배 증가합니다
            - 큰 이미지의 경우 처리 시간이 오래 걸릴 수 있습니다
            - 단순 확대만으로는 원본보다 더 선명해지지 않습니다
            """)
        
        # 이벤트 핸들러
        def process_upscale(img, method, enhance):
            if img is None:
                return None, None, "이미지를 업로드해주세요."
            
            try:
                original, upscaled, info = upscale_image_4x(img, method, enhance)
                return original, upscaled, info
            except Exception as e:
                return None, None, f"오류가 발생했습니다: {str(e)}"
        
        def process_compare(img):
            if img is None:
                return None, None, None, None, None
            
            try:
                return compare_methods(img)
            except Exception as e:
                return None, None, None, None, None
        
        def save_image(upscaled_img):
            if upscaled_img is None:
                return None
            
            try:
                file_path = save_upscaled_image(upscaled_img)
                return file_path
            except Exception as e:
                return None
        
        # 이벤트 연결
        upscale_btn.click(
            fn=process_upscale,
            inputs=[input_image, interpolation_method, enhance_edges],
            outputs=[original_output, upscaled_output, info_output]
        )
        
        compare_btn.click(
            fn=process_compare,
            inputs=[compare_input],
            outputs=[compare_original, compare_nearest, compare_bilinear, compare_cubic, compare_lanczos]
        )
        
        # 다운로드 버튼은 upscaled_output이 변경될 때마다 업데이트
        upscaled_output.change(
            fn=save_image,
            inputs=[upscaled_output],
            outputs=[download_btn]
        )
    
    return interface

# 메인 실행
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 