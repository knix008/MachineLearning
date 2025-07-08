# OpenCV C++ 4배 이미지 확대 코드

이 프로젝트는 OpenCV C++를 사용하여 이미지를 4배 확대하는 다양한 방법을 제공합니다.

## 파일 설명

### 1. `simple_upscale_4x.cpp` (추천)
- **간단하고 실용적인** 4배 이미지 확대 코드
- 기본적인 OpenCV 기능만 사용
- 20줄 이내의 핵심 코드

### 2. `upscale_4x.cpp`
- **고급 기능이 포함된** 확장 버전
- 다양한 보간법 비교 기능
- 엣지 강화 적용 기능
- 이미지 정보 출력 기능

## 설치 요구사항

### Windows (Visual Studio)
```bash
# vcpkg 사용 (권장)
vcpkg install opencv4[core,imgcodecs,imgproc]

# 또는 직접 다운로드
# https://opencv.org/releases/ 에서 다운로드
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install libopencv-dev
```

### macOS
```bash
# Homebrew 사용
brew install opencv
```

## 컴파일 및 실행

### CMake 사용 (권장)
```bash
# 빌드 디렉토리 생성
mkdir build
cd build

# CMake 설정
cmake ..

# 컴파일
make

# 실행
./simple_upscale_4x
./upscale_4x
```

### 직접 컴파일
```bash
# 간단한 버전
g++ -std=c++11 simple_upscale_4x.cpp -o simple_upscale_4x `pkg-config --cflags --libs opencv4`

# 고급 버전
g++ -std=c++11 upscale_4x.cpp -o upscale_4x `pkg-config --cflags --libs opencv4`
```

## 사용법

### 기본 사용법
```cpp
#include <opencv2/opencv.hpp>

int main() {
    // 이미지 읽기
    cv::Mat image = cv::imread("input.jpg");
    
    // 4배 확대
    cv::Mat upscaled;
    cv::resize(image, upscaled, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);
    
    // 결과 저장
    cv::imwrite("output.jpg", upscaled);
    
    return 0;
}
```

### 다양한 보간법 사용
```cpp
// Nearest Neighbor (가장 빠름, 품질 낮음)
cv::resize(image, upscaled, cv::Size(), 4.0, 4.0, cv::INTER_NEAREST);

// Bilinear (적당한 속도와 품질)
cv::resize(image, upscaled, cv::Size(), 4.0, 4.0, cv::INTER_LINEAR);

// Bicubic (높은 품질, 기본값)
cv::resize(image, upscaled, cv::Size(), 4.0, 4.0, cv::INTER_CUBIC);

// Lanczos (최고 품질, 느림)
cv::resize(image, upscaled, cv::Size(), 4.0, 4.0, cv::INTER_LANCZOS4);
```

## 보간법 종류

1. **INTER_NEAREST**
   - 가장 빠르지만 품질이 낮음
   - 픽셀화 현상 발생

2. **INTER_LINEAR**
   - 적당한 속도와 품질
   - 일반적인 용도에 적합

3. **INTER_CUBIC**
   - 높은 품질 (기본값)
   - 대부분의 경우에 권장

4. **INTER_LANCZOS4**
   - 최고 품질
   - 처리 시간이 오래 걸림

## 출력 파일

- `upscaled_4x.jpg`: 간단한 버전의 결과
- `upscaled_4x_basic.jpg`: 기본 4배 확대 결과
- `upscaled_4x_enhanced.jpg`: 엣지 강화 적용 결과
- `output/upscaled_4x_nearest.jpg`: Nearest Neighbor 방법
- `output/upscaled_4x_bilinear.jpg`: Bilinear 방법
- `output/upscaled_4x_bicubic.jpg`: Bicubic 방법
- `output/upscaled_4x_lanczos.jpg`: Lanczos 방법

## 고급 기능

### 엣지 강화 과정
1. Bicubic 보간법으로 4배 확대
2. 언샤프 마스킹으로 엣지 강화
3. 노이즈 제거

### 이미지 정보 출력
- 크기 (너비 x 높이)
- 채널 수
- 데이터 타입
- 메모리 사용량

## 성능 비교

| 보간법 | 속도 | 품질 | 메모리 사용량 |
|--------|------|------|---------------|
| INTER_NEAREST | 매우 빠름 | 낮음 | 낮음 |
| INTER_LINEAR | 빠름 | 보통 | 보통 |
| INTER_CUBIC | 보통 | 높음 | 높음 |
| INTER_LANCZOS4 | 느림 | 매우 높음 | 매우 높음 |

## 주의사항

1. **메모리 사용량**: 4배 확대시 메모리 사용량이 16배 증가합니다
2. **처리 시간**: 이미지 크기가 클수록 처리 시간이 오래 걸립니다
3. **품질**: 단순 확대만으로는 원본보다 더 선명해지지 않습니다
4. **파일 경로**: Windows에서는 백슬래시(`\`) 대신 슬래시(`/`) 사용 권장

## 문제 해결

### 컴파일 오류
```bash
# OpenCV가 설치되지 않은 경우
sudo apt install libopencv-dev  # Ubuntu/Debian
brew install opencv             # macOS
```

### 이미지를 읽을 수 없는 경우
- 파일 경로가 올바른지 확인
- 이미지 파일이 손상되지 않았는지 확인
- 지원하는 이미지 형식인지 확인 (jpg, png, bmp 등)

### 메모리 부족 오류
- 더 작은 이미지로 테스트
- 배치 처리 방식 사용 고려

## 라이선스

이 코드는 MIT 라이선스 하에 배포됩니다. 