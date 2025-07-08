import gradio as gr
import cv2
import numpy as np

def upscale_4x_simple(image, method="cubic"):
    """
    간단한 4배 이미지 확대 함수
    
    Args:
        image: PIL Image
        method: 보간법 ('nearest', 'bilinear', 'cubic', 'lanczos')
    
    Returns:
        tuple: (원본, 확대된 이미지)
    """
    
    if image is None:
        return None, None
    
    # PIL Image를 numpy array로 변환
    image_np = np.array(image)
    
    # RGB를 BGR로 변환 (OpenCV용)
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 보간법 매핑
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interpolation = interpolation_map.get(method, cv2.INTER_CUBIC)
    
    # 4배 확대
    upscaled = cv2.resize(image_np, None, fx=4.0, fy=4.0, interpolation=interpolation)
    
    # BGR을 RGB로 변환 (PIL용)
    if len(upscaled.shape) == 3:
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    else:
        upscaled_rgb = upscaled
    
    # 원본 이미지를 RGB로 변환
    if len(image_np.shape) == 3:
        original_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = image_np
    
    return original_rgb, upscaled_rgb

# Gradio 인터페이스
def create_simple_interface():
    with gr.Interface(
        fn=upscale_4x_simple,
        inputs=[
            gr.Image(label="원본 이미지", type="pil"),
            gr.Dropdown(
                choices=["cubic", "bilinear", "nearest", "lanczos"],
                value="cubic",
                label="보간법",
                info="cubic: 높은 품질, bilinear: 적당한 품질, nearest: 빠름, lanczos: 최고 품질"
            )
        ],
        outputs=[
            gr.Image(label="원본", height=300),
            gr.Image(label="4배 확대", height=300)
        ],
        title="🔍 OpenCV 4배 이미지 확대",
        description="이미지를 4배 확대하는 간단한 도구입니다.",
        examples=[
            ["images/Lenna.png", "cubic"],
            ["images/Lenna.png", "bilinear"],
            ["images/Lenna.png", "nearest"],
            ["images/Lenna.png", "lanczos"]
        ]
    ) as interface:
        
        gr.Markdown("""
        ## 사용법
        1. 이미지를 업로드하거나 드래그 앤 드롭하세요
        2. 보간법을 선택하세요
        3. 결과를 확인하세요
        
        ## 보간법 설명
        - **Cubic**: 높은 품질 (권장)
        - **Bilinear**: 적당한 속도와 품질
        - **Nearest**: 가장 빠르지만 품질이 낮음
        - **Lanczos**: 최고 품질, 처리 시간이 오래 걸림
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_simple_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    ) 