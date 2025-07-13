# Stable Diffusion 3.5 스타일 보존 이미지 생성

이 프로그램은 Stable Diffusion 3.5 Medium을 사용하여 입력 이미지의 스타일을 유지하면서 새로운 이미지를 생성하는 웹 애플리케이션입니다.

## 🎯 주요 기능

- **스타일 보존**: 입력 이미지의 아트 스타일과 구도를 유지
- **프롬프트 기반 생성**: 텍스트 프롬프트에 따라 새로운 내용 생성
- **파라미터 조정**: 변환 강도, 가이던스 스케일, 추론 스텝 등 세밀한 조정 가능
- **네거티브 프롬프트**: 원하지 않는 요소들을 제거
- **랜덤 시드**: 재현 가능한 결과를 위한 시드 설정
- **사용자 친화적 인터페이스**: Gradio 기반의 직관적인 웹 UI

## 🚀 설치 및 실행

### 1. 환경 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장) 또는 CPU
- 최소 8GB RAM (GPU 사용시 12GB 이상 권장)
- CPU 사용시: 최소 8GB RAM, 4코어 이상 권장

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 프로그램 실행

#### Windows
```bash
run_style_preserving_generation.bat
```

#### Linux/Mac
```bash
python stablediffusion-3.5-style-transfer03.py
```

### 4. 웹 브라우저 접속

프로그램 실행 후 웹 브라우저에서 `http://localhost:7860`으로 접속하세요.

### 5. CPU 사용시 주의사항

- **생성 시간**: CPU에서는 GPU보다 훨씬 오래 걸립니다 (5-15분)
- **메모리 사용량**: 최소 8GB RAM이 필요하며, 더 많은 RAM이 권장됩니다
- **이미지 크기**: 512x512 이하로 설정하는 것을 권장합니다
- **최적화**: CPU 최적화 옵션을 활성화하면 안정성이 향상됩니다

## 📖 사용법

### 1. 이미지 업로드
- 스타일을 참조할 이미지를 업로드합니다
- 지원 형식: JPG, PNG, WebP
- 권장 크기: 512x512 ~ 768x768 픽셀

### 2. 프롬프트 입력
- 생성하고 싶은 이미지를 설명하는 프롬프트를 입력합니다
- 예시: "a beautiful landscape with mountains and lake"

### 3. 네거티브 프롬프트 (선택사항)
- 원하지 않는 요소들을 제거하기 위한 프롬프트
- 예시: "blurry, low quality, distorted"

### 4. 파라미터 조정

#### 변환 강도 (Strength)
- **범위**: 0.1 ~ 1.0
- **권장값**: 0.4 ~ 0.7
- **설명**: 높을수록 원본 이미지에서 더 많이 벗어납니다

#### 가이던스 스케일 (Guidance Scale)
- **범위**: 1.0 ~ 20.0
- **권장값**: 7.0 ~ 10.0
- **설명**: 높을수록 프롬프트를 더 정확히 따릅니다

#### 추론 스텝 수 (Inference Steps)
- **범위**: 10 ~ 50
- **권장값**: 20 ~ 30
- **설명**: 높을수록 품질이 좋아지지만 시간이 오래 걸립니다

### 5. 스타일 보존 설정

#### 스타일 보존 활성화
- 체크박스로 스타일 보존 기능을 켜고 끌 수 있습니다
- 활성화시 원본 이미지의 아트 스타일을 유지합니다

#### 스타일 보존 강도
- **범위**: 0.0 ~ 1.0
- **권장값**: 0.7 ~ 0.9
- **설명**: 스타일을 얼마나 강하게 보존할지 설정합니다

### 6. CPU 최적화 설정

#### CPU 최적화 활성화
- CPU 사용시 메모리 사용량을 줄이고 안정성을 높입니다
- 자동으로 추론 스텝과 가이던스 스케일을 조정합니다

#### 최대 이미지 크기
- **GPU 권장**: 768x768
- **CPU 권장**: 512x512
- **설명**: CPU 사용시 더 작은 크기를 권장합니다

### 7. 이미지 생성
"🎨 이미지 생성하기" 버튼을 클릭하여 이미지를 생성합니다.

## 💡 프롬프트 작성 팁

### 좋은 프롬프트 예시
```
"a futuristic city with flying cars, neon lights, cyberpunk style"
"a magical forest with glowing mushrooms, fantasy art style"
"a cozy coffee shop interior, warm lighting, vintage style"
```

### 네거티브 프롬프트 예시
```
"blurry, low quality, distorted, ugly, watermark"
"oversaturated, overexposed, underexposed"
"deformed, disfigured, bad anatomy"
```

## ⚙️ 고급 설정

### 랜덤 시드
- -1: 랜덤 시드 사용
- 양수: 특정 시드 사용 (재현 가능한 결과)

### 메모리 최적화
- 큰 이미지는 자동으로 설정된 최대 크기 이하로 리사이즈됩니다
- GPU 메모리 부족시 CPU 모드로 자동 전환됩니다
- CPU 사용시 메모리 사용량을 최소화하는 최적화가 적용됩니다

## 🔧 문제 해결

### 일반적인 문제들

#### 1. CUDA 메모리 부족
```
오류: CUDA out of memory
해결책: 
- 이미지 크기를 줄이세요
- 추론 스텝 수를 줄이세요
- CPU 모드를 사용하세요
```

#### 2. CPU 메모리 부족
```
오류: CPU out of memory 또는 시스템이 느려짐
해결책:
- 이미지 크기를 512x512 이하로 설정하세요
- CPU 최적화 옵션을 활성화하세요
- 다른 프로그램을 종료하여 메모리를 확보하세요
```

#### 3. 모델 다운로드 실패
```
오류: Connection error during model download
해결책:
- 인터넷 연결을 확인하세요
- Hugging Face 토큰을 설정하세요
- 프록시 설정을 확인하세요
```

#### 4. 생성 품질이 낮음
```
해결책:
- 추론 스텝 수를 늘리세요 (25-30)
- 가이던스 스케일을 조정하세요 (7-10)
- 더 구체적인 프롬프트를 사용하세요
```

#### 5. CPU에서 생성이 너무 느림
```
해결책:
- 이미지 크기를 512x512 이하로 설정하세요
- 추론 스텝 수를 20 이하로 줄이세요
- CPU 최적화 옵션을 활성화하세요
- GPU가 있다면 GPU 사용을 권장합니다
```

## 📁 파일 구조

```
StableDiffusion/
├── stablediffusion-3.5-style-transfer03.py  # 메인 프로그램
├── run_style_preserving_generation.bat      # Windows 실행 파일
├── requirements.txt                          # 필요한 패키지 목록
└── README_style_preserving_generation.md    # 이 파일
```

## 🤝 기여하기

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- Stability AI의 Stable Diffusion 3.5 Medium 모델
- Hugging Face의 diffusers 라이브러리
- Gradio 팀의 웹 인터페이스 프레임워크 