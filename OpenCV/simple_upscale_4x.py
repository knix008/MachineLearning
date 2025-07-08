import cv2
import numpy as np

def upscale_4x_simple(image_path, output_path=None):
    """
    간단한 4배 이미지 확대 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로 (None이면 저장하지 않음)
    
    Returns:
        numpy.ndarray: 확대된 이미지
    """
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    # 원본 크기 출력
    height, width = image.shape[:2]
    print(f"원본 크기: {width} x {height}")
    
    # 4배 확대 (Bicubic 보간법 사용)
    upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 확대된 크기 출력
    new_height, new_width = upscaled.shape[:2]
    print(f"확대된 크기: {new_width} x {new_height}")
    
    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, upscaled)
        print(f"확대된 이미지가 저장되었습니다: {output_path}")
    
    return upscaled

def upscale_4x_with_methods(image_path, output_dir="output"):
    """
    다양한 보간법으로 4배 확대하는 함수
    
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
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return
    
    # 보간법들
    methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    # 각 방법으로 확대
    for method_name, interpolation in methods.items():
        print(f"{method_name} 방법으로 확대 중...")
        
        # 4배 확대
        upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=interpolation)
        
        # 결과 저장
        output_path = os.path.join(output_dir, f"upscaled_4x_{method_name}.jpg")
        cv2.imwrite(output_path, upscaled)
        print(f"저장됨: {output_path}")

def upscale_4x_enhanced(image_path, output_path=None):
    """
    엣지 강화를 적용한 4배 확대 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로
    
    Returns:
        numpy.ndarray: 확대된 이미지
    """
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    # 1단계: Bicubic 보간법으로 4배 확대
    upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 2단계: 언샤프 마스킹으로 엣지 강화
    blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    
    # 3단계: 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # 결과 저장
    if output_path:
        cv2.imwrite(output_path, denoised)
        print(f"엣지 강화된 확대 이미지가 저장되었습니다: {output_path}")
    
    return denoised

# 사용 예제
if __name__ == "__main__":
    # 테스트 이미지 경로
    test_image = "images/Lenna.png"
    
    print("=== OpenCV 4배 이미지 확대 ===\n")
    
    # 1. 기본 4배 확대
    print("1. 기본 4배 확대")
    result1 = upscale_4x_simple(test_image, "upscaled_4x_basic.jpg")
    print()
    
    # 2. 다양한 보간법 비교
    print("2. 다양한 보간법 비교")
    upscale_4x_with_methods(test_image)
    print()
    
    # 3. 엣지 강화 적용
    print("3. 엣지 강화 적용")
    result3 = upscale_4x_enhanced(test_image, "upscaled_4x_enhanced.jpg")
    print()
    
    print("모든 처리가 완료되었습니다!") 