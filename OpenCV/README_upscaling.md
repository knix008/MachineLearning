# OpenCV 4배 이미지 확대 코드

이 프로젝트는 OpenCV를 사용하여 이미지를 4배 확대하는 다양한 방법을 제공합니다.

## 파일 설명

### 1. `simple_upscale_4x.py` (추천)
- 간단하고 실용적인 4배 이미지 확대 코드
- 기본적인 OpenCV 기능만 사용
- 다양한 보간법 비교 기능 포함

### 2. `image_upscaling_4x.py`
- 고급 기능이 포함된 확장 버전
- matplotlib을 사용한 시각화 기능
- 더 상세한 분석 기능

## 설치 요구사항

```bash
pip install opencv-python
pip install numpy
pip install matplotlib  # 시각화 기능 사용시에만 필요
```

## 사용법

### 기본 사용법

```python
from simple_upscale_4x import upscale_4x_simple

# 기본 4배 확대
result = upscale_4x_simple("input_image.jpg", "output_image.jpg")
```

### 다양한 보간법 비교

```python
from simple_upscale_4x import upscale_4x_with_methods

# 4가지 보간법으로 확대하여 비교
upscale_4x_with_methods("input_image.jpg", "output_folder")
```

### 엣지 강화 적용

```python
from simple_upscale_4x import upscale_4x_enhanced

# 엣지 강화를 적용한 4배 확대
result = upscale_4x_enhanced("input_image.jpg", "enhanced_output.jpg")
```

## 보간법 종류

1. **Nearest Neighbor (INTER_NEAREST)**
   - 가장 빠르지만 품질이 낮음
   - 픽셀화 현상 발생

2. **Bilinear (INTER_LINEAR)**
   - 적당한 속도와 품질
   - 일반적인 용도에 적합

3. **Bicubic (INTER_CUBIC)**
   - 높은 품질 (기본값)
   - 대부분의 경우에 권장

4. **Lanczos (INTER_LANCZOS4)**
   - 최고 품질
   - 처리 시간이 오래 걸림

## 실행 예제

```bash
# 기본 실행
python simple_upscale_4x.py

# 고급 기능 실행
python image_upscaling_4x.py
```

## 출력 파일

- `upscaled_4x_basic.jpg`: 기본 4배 확대 결과
- `upscaled_4x_enhanced.jpg`: 엣지 강화 적용 결과
- `output/upscaled_4x_nearest.jpg`: Nearest Neighbor 방법
- `output/upscaled_4x_bilinear.jpg`: Bilinear 방법
- `output/upscaled_4x_bicubic.jpg`: Bicubic 방법
- `output/upscaled_4x_lanczos.jpg`: Lanczos 방법

## 주의사항

1. **메모리 사용량**: 4배 확대시 메모리 사용량이 16배 증가합니다
2. **처리 시간**: 이미지 크기가 클수록 처리 시간이 오래 걸립니다
3. **품질**: 단순 확대만으로는 원본보다 더 선명해지지 않습니다

## 고급 기능

### 엣지 강화 과정
1. Bicubic 보간법으로 4배 확대
2. 언샤프 마스킹으로 엣지 강화
3. 노이즈 제거

### 시각화 기능
- 원본과 확대 결과 비교
- 다양한 보간법 결과 비교
- matplotlib을 사용한 그래프 생성

## 문제 해결

### 이미지를 읽을 수 없는 경우
- 파일 경로가 올바른지 확인
- 이미지 파일이 손상되지 않았는지 확인
- 지원하는 이미지 형식인지 확인 (jpg, png, bmp 등)

### 메모리 부족 오류
- 더 작은 이미지로 테스트
- 배치 처리 방식 사용 고려

## 라이선스

이 코드는 MIT 라이선스 하에 배포됩니다. 