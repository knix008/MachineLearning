# Stable Diffusion 3.5 Medium 이미지 스타일 변환

Stable Diffusion 3.5 Medium을 사용하여 입력 이미지를 원하는 스타일로 변환하는 Gradio 웹 애플리케이션입니다.

## 🚀 주요 기능

- **이미지-투-이미지 변환**: 기존 이미지를 새로운 스타일로 변환
- **다양한 스타일 지원**: 유화화, 수채화, 애니메, 스케치 등 다양한 스타일
- **직관적인 UI**: Gradio를 통한 사용자 친화적 인터페이스
- **파라미터 조정**: 변환 강도, 가이던스 스케일, 추론 스텝 수 등 세밀한 조정 가능
- **실시간 미리보기**: 변환 결과를 즉시 확인

## 📋 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (권장, CPU도 사용 가능)
- 최소 8GB RAM (GPU 사용시 12GB 이상 권장)

## 🔧 설치 방법

1. **의존성 패키지 설치**:
```bash
pip install -r requirements.txt
```

2. **GPU 지원 확인** (선택사항):
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🎯 사용 방법

1. **애플리케이션 실행**:
```bash
python stablediffusion-3.5-style-transfer-gradio.py
```

2. **웹 브라우저에서 접속**:
   - 로컬: `http://localhost:7860`
   - 공개 링크: 터미널에 표시되는 공개 URL

3. **이미지 변환 과정**:
   - 변환할 이미지 업로드
   - 원하는 스타일 프롬프트 입력
   - 파라미터 조정 (선택사항)
   - "이미지 변환하기" 버튼 클릭

## 🎨 스타일 프롬프트 예시

### 기본 스타일
- `oil painting, masterpiece, detailed, vibrant colors` - 유화화 스타일
- `watercolor painting, soft, dreamy, artistic` - 수채화 스타일
- `anime style, cel shading, vibrant, detailed` - 애니메 스타일
- `photorealistic, cinematic lighting, professional photography` - 사진같은 스타일

### 특수 스타일
- `sketch, pencil drawing, black and white, artistic` - 스케치 스타일
- `impressionist painting, brush strokes, colorful` - 인상주의 스타일
- `cyberpunk style, neon lights, futuristic` - 사이버펑크 스타일
- `vintage, retro, 1950s style, nostalgic` - 빈티지 스타일

## ⚙️ 파라미터 설명

### 변환 강도 (Strength)
- **범위**: 0.1 - 1.0
- **권장값**: 0.3 - 0.7
- **설명**: 높을수록 원본 이미지에서 더 많이 벗어나 변환됩니다

### 가이던스 스케일 (Guidance Scale)
- **범위**: 1.0 - 20.0
- **권장값**: 7.0 - 10.0
- **설명**: 높을수록 프롬프트를 더 정확히 따릅니다

### 추론 스텝 수 (Inference Steps)
- **범위**: 10 - 50
- **권장값**: 20 - 30
- **설명**: 높을수록 품질이 좋아지지만 시간이 오래 걸립니다

## 💡 사용 팁

1. **프롬프트 작성**:
   - 구체적이고 상세한 설명이 좋은 결과를 만듭니다
   - 스타일 + 품질 키워드를 조합하세요 (예: "oil painting, masterpiece, detailed")

2. **네거티브 프롬프트**:
   - 원하지 않는 요소를 제거하는데 도움이 됩니다
   - 예: "blurry, low quality, distorted, ugly"

3. **이미지 크기**:
   - 너무 큰 이미지는 메모리 사용량을 증가시킵니다
   - 768x768 이하로 자동 조정됩니다

4. **시드 값**:
   - 같은 시드를 사용하면 동일한 결과를 얻을 수 있습니다
   - -1로 설정하면 랜덤 시드가 사용됩니다

## 🔍 문제 해결

### 메모리 부족 오류
- 이미지 크기를 줄여보세요
- 추론 스텝 수를 줄여보세요
- GPU 메모리가 부족한 경우 CPU 모드로 전환

### 느린 처리 속도
- 추론 스텝 수를 줄여보세요
- GPU 사용을 확인하세요
- 다른 프로그램을 종료하여 메모리를 확보하세요

### 품질이 좋지 않은 결과
- 프롬프트를 더 구체적으로 작성해보세요
- 가이던스 스케일을 높여보세요
- 추론 스텝 수를 늘려보세요

## 📝 라이선스

이 프로젝트는 Stable Diffusion 3.5 Medium 모델을 사용합니다. 모델 사용에 대한 라이선스는 Stability AI의 정책을 따릅니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해주세요.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 통해 문의해주세요. 