import cv2

def upscale_4x(image_path, output_path):
    """
    이미지를 4배 확대하는 간단한 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str): 출력 이미지 경로
    """
    
    # 이미지 읽기
    image = cv2.imread(image_path)
    
    # 4배 확대 (Bicubic 보간법 사용)
    upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # 결과 저장
    cv2.imwrite(output_path, upscaled)
    
    print(f"이미지가 4배 확대되어 {output_path}에 저장되었습니다.")

# 사용 예제
if __name__ == "__main__":
    # 이미지 확대
    upscale_4x("images/Lenna.png", "upscaled_4x.jpg") 