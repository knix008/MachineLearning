# Gradio OpenCV 4배 이미지 확대 GUI

이 프로젝트는 Gradio를 사용하여 OpenCV 4배 이미지 확대 기능을 웹 기반 GUI로 제공합니다.

## 파일 설명

### 1. `simple_gradio_gui.py` (추천)
- **간단하고 직관적인** Gradio 인터페이스
- 기본적인 4배 확대 기능
- 예제 이미지 포함
- 빠르게 시작할 수 있는 버전

### 2. `gradio_upscale_gui.py`
- **고급 기능이 포함된** 확장 버전
- 탭 기반 인터페이스
- 보간법 비교 기능
- 엣지 강화 옵션
- 이미지 다운로드 기능
- 상세한 처리 정보

## 설치 요구사항

```bash
# 기본 패키지 설치
pip install gradio
pip install opencv-python
pip install numpy
pip install pillow

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### requirements.txt
```
gradio>=4.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
```

## 실행 방법

### 1. 간단한 버전 실행
```bash
python simple_gradio_gui.py
```

### 2. 고급 버전 실행
```bash
python gradio_upscale_gui.py
```

## GUI 기능

### 간단한 버전 (`simple_gradio_gui.py`)

#### 주요 기능:
- **이미지 업로드**: 드래그 앤 드롭 또는 파일 선택
- **보간법 선택**: 4가지 보간법 중 선택
- **실시간 미리보기**: 원본과 확대 결과를 나란히 표시
- **예제 제공**: 미리 정의된 예제로 빠른 테스트

#### 사용법:
1. 이미지를 업로드하거나 드래그 앤 드롭
2. 보간법 선택 (cubic, bilinear, nearest, lanczos)
3. 결과 확인

### 고급 버전 (`gradio_upscale_gui.py`)

#### 탭 구성:

##### 📤 기본 확대 탭
- **이미지 업로드**: 원본 이미지 선택
- **설정 옵션**:
  - 보간법 선택 (cubic, bilinear, nearest, lanczos)
  - 엣지 강화 체크박스
- **결과 표시**:
  - 원본 이미지
  - 4배 확대된 이미지
  - 처리 정보 (크기, 메모리 사용량 등)
- **다운로드**: 확대된 이미지 다운로드

##### 🔄 보간법 비교 탭
- **이미지 업로드**: 원본 이미지 선택
- **비교 결과**: 5개 이미지 동시 표시
  - 원본
  - Nearest Neighbor
  - Bilinear
  - Bicubic
  - Lanczos

##### 📚 정보 탭
- **사용법 가이드**: 단계별 사용법
- **보간법 설명**: 각 방법의 특징
- **주의사항**: 메모리 사용량, 처리 시간 등

## 웹 인터페이스 접속

실행 후 다음 URL로 접속:

### 로컬 접속
```
http://localhost:7860
```

### 공개 링크 (share=True 설정시)
```
https://xxxxx.gradio.live
```

## 주요 특징

### 🎨 사용자 친화적 인터페이스
- 직관적인 드래그 앤 드롭
- 실시간 미리보기
- 반응형 디자인

### ⚡ 빠른 처리
- OpenCV 기반 고성능 처리
- 다양한 보간법 지원
- 실시간 결과 표시

### 📊 상세한 정보
- 이미지 크기 정보
- 메모리 사용량 계산
- 처리 시간 표시

### 💾 다운로드 기능
- 확대된 이미지 자동 저장
- 임시 파일 관리
- 다양한 형식 지원

## 보간법 비교

| 보간법 | 속도 | 품질 | 용도 |
|--------|------|------|------|
| **Nearest** | 매우 빠름 | 낮음 | 빠른 미리보기 |
| **Bilinear** | 빠름 | 보통 | 일반적인 용도 |
| **Cubic** | 보통 | 높음 | 권장 (기본값) |
| **Lanczos** | 느림 | 매우 높음 | 고품질 요구시 |

## 사용 예제

### 1. 기본 사용
```python
# 간단한 버전 실행
python simple_gradio_gui.py
```

### 2. 고급 기능 사용
```python
# 고급 버전 실행
python gradio_upscale_gui.py
```

### 3. 커스텀 설정
```python
# 포트 변경
interface.launch(server_port=8080)

# 공개 링크 비활성화
interface.launch(share=False)

# 디버그 모드
interface.launch(debug=True)
```

## 문제 해결

### Gradio 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# Gradio 재설치
pip uninstall gradio
pip install gradio
```

### OpenCV 오류
```bash
# OpenCV 재설치
pip uninstall opencv-python
pip install opencv-python-headless
```

### 포트 충돌
```bash
# 다른 포트 사용
python simple_gradio_gui.py --port 8080
```

### 메모리 부족
- 더 작은 이미지 사용
- 배치 처리 고려
- 시스템 메모리 확인

## 성능 최적화

### 1. 이미지 크기 제한
```python
# 최대 이미지 크기 설정
gr.Image(max_size=1024)
```

### 2. 캐싱 활성화
```python
# 함수 캐싱
@gr.cache
def upscale_function(image, method):
    # 처리 로직
    pass
```

### 3. 배치 처리
```python
# 여러 이미지 동시 처리
def batch_upscale(images, method):
    results = []
    for img in images:
        result = upscale_4x(img, method)
        results.append(result)
    return results
```

## 라이선스

이 코드는 MIT 라이선스 하에 배포됩니다.

## 기여하기

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 