import cv2
import numpy as np
import matplotlib.pyplot as plt

def upscale_image_4x(image_path, output_path=None, interpolation_method='cubic'):
    """
    OpenCV를 사용하여 이미지를 4배 확대하는 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로 (None이면 저장하지 않음)
        interpolation_method (str): 보간법 ('nearest', 'bilinear', 'cubic', 'lanczos')
    
    Returns:
        numpy.ndarray: 확대된 이미지
    """
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    # 원본 이미지 크기
    height, width = image.shape[:2]
    print(f"원본 이미지 크기: {width} x {height}")
    
    # 보간법 매핑
    interpolation_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    if interpolation_method not in interpolation_map:
        raise ValueError(f"지원하지 않는 보간법: {interpolation_method}")
    
    # 4배 확대
    new_width = width * 4
    new_height = height * 4
    
    # 이미지 리사이즈
    upscaled_image = cv2.resize(
        image, 
        (new_width, new_height), 
        interpolation=interpolation_map[interpolation_method]
    )
    
    print(f"확대된 이미지 크기: {new_width} x {new_height}")
    
    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, upscaled_image)
        print(f"확대된 이미지가 저장되었습니다: {output_path}")
    
    return upscaled_image

def compare_upscaling_methods(image_path, output_dir="upscaled_images"):
    """
    다양한 보간법을 사용하여 이미지 확대 결과를 비교하는 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_dir (str): 출력 디렉토리
    """
    
    import os
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    # BGR을 RGB로 변환 (matplotlib용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 보간법들
    methods = {
        'Nearest Neighbor': cv2.INTER_NEAREST,
        'Bilinear': cv2.INTER_LINEAR,
        'Bicubic': cv2.INTER_CUBIC,
        'Lanczos': cv2.INTER_LANCZOS4
    }
    
    # 결과 저장을 위한 리스트
    results = []
    
    # 각 방법으로 확대
    for method_name, interpolation in methods.items():
        print(f"{method_name} 방법으로 확대 중...")
        
        # 4배 확대
        upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=interpolation)
        
        # BGR을 RGB로 변환
        upscaled_rgb = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
        
        # 결과 저장
        output_path = os.path.join(output_dir, f"upscaled_4x_{method_name.lower().replace(' ', '_')}.jpg")
        cv2.imwrite(output_path, upscaled)
        
        results.append((method_name, upscaled_rgb))
        print(f"저장됨: {output_path}")
    
    # 결과 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 원본 이미지
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('원본 이미지')
    axes[0, 0].axis('off')
    
    # 확대된 이미지들
    for i, (method_name, upscaled_img) in enumerate(results):
        row = (i + 1) // 2
        col = (i + 1) % 2 + 1
        axes[row, col].imshow(upscaled_img)
        axes[row, col].set_title(f'{method_name} (4x)')
        axes[row, col].axis('off')
    
    # 마지막 subplot 숨기기
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def upscale_with_edge_enhancement(image_path, output_path=None):
    """
    엣지 강화를 적용하여 이미지를 4배 확대하는 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로
    
    Returns:
        numpy.ndarray: 확대된 이미지
    """
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    
    # 1단계: Bicubic 보간법으로 4배 확대
    upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 2단계: 언샤프 마스킹으로 엣지 강화
    # 가우시안 블러 적용
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
    
    # 언샤프 마스킹
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    
    # 3단계: 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, denoised)
        print(f"엣지 강화된 확대 이미지가 저장되었습니다: {output_path}")
    
    return denoised

if __name__ == "__main__":
    # 예제 실행
    try:
        # 테스트 이미지 경로 (images 폴더에 있는 이미지 사용)
        test_image_path = "images/Lenna.png"
        
        print("=== OpenCV 4배 이미지 확대 예제 ===\n")
        
        # 1. 기본 4배 확대 (Bicubic 보간법)
        print("1. 기본 4배 확대 (Bicubic 보간법)")
        upscaled_basic = upscale_image_4x(
            test_image_path, 
            "upscaled_4x_basic.jpg", 
            interpolation_method='cubic'
        )
        print()
        
        # 2. 엣지 강화를 적용한 4배 확대
        print("2. 엣지 강화를 적용한 4배 확대")
        upscaled_enhanced = upscale_with_edge_enhancement(
            test_image_path, 
            "upscaled_4x_enhanced.jpg"
        )
        print()
        
        # 3. 다양한 보간법 비교
        print("3. 다양한 보간법 비교")
        compare_upscaling_methods(test_image_path)
        print()
        
        print("모든 처리가 완료되었습니다!")
        
    except FileNotFoundError:
        print(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")
        print("다른 이미지 파일 경로를 지정하거나 images 폴더에 이미지를 추가해주세요.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}") 