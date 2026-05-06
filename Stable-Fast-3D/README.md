# Stable Fast 3D

단일 이미지에서 UV 텍스처와 조명 분리가 적용된 3D 메시를 빠르게 생성합니다.  
원본 모델: [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)

---

## 주요 기능

- 단일 이미지 → GLB(3D 메시 + 텍스처) 변환
- CUDA 자동 감지 — GPU 없으면 CPU로 자동 전환
- 배경 자동 제거 (rembg)
- 리메시(triangle / quad) 및 버텍스 수 제한 지원
- Gradio 웹 UI, CLI, ComfyUI 노드 세 가지 인터페이스 제공

---

## 요구 사항

- Python 3.10 이상
- PyTorch (CUDA 또는 CPU 버전)

---

## 설치

### 1. PyTorch 설치 (먼저 실행)

**CUDA 12.6 (GPU 사용 시):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU 전용:**
```bash
pip install torch torchvision
```

### 2. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

> texture_baker / uv_unwrapper 빌드가 실패하면:
> ```bash
> pip install -r requirements.txt --no-build-isolation
> ```

### 3. Gradio UI 추가 패키지 (웹 앱 사용 시)

```bash
pip install -r requirements-demo.txt
```

---

## 모델 가중치

최초 실행 시 Hugging Face에서 자동 다운로드됩니다.  
수동 다운로드 또는 캐시 경로 지정:

```python
model = SF3D.from_pretrained("stabilityai/stable-fast-3d", ...)
# 로컬 경로 사용 시:
model = SF3D.from_pretrained("/path/to/model", ...)
```

---

## 사용법

### 1. 커맨드라인 (run.py)

```bash
# 단일 이미지
python run.py demo_files/examples/chair1.png --output-dir output/

# 여러 이미지
python run.py img1.png img2.png --output-dir output/

# 폴더 전체
python run.py demo_files/examples/ --output-dir output/

# 주요 옵션
python run.py image.png \
    --device cuda \          # cuda / cpu (기본: 자동 감지)
    --texture-resolution 1024 \
    --remesh_option triangle \
    --target_vertex_count 5000 \
    --foreground-ratio 0.85 \
    --batch_size 1
```

### 2. 간편 래퍼 (convert_2d_to_3d.py)

run.py와 동일한 인수를 그대로 사용합니다.

```bash
python convert_2d_to_3d.py demo_files/examples/chair1.png --output-dir output/
```

### 3. Gradio 웹 UI

```bash
python gradio_app.py
```

브라우저에서 `http://localhost:7860` 접속 후 이미지를 업로드하면 3D 모델을 생성할 수 있습니다.

### 4. ComfyUI 노드

이 디렉토리를 ComfyUI의 `custom_nodes/` 폴더 안에 놓으면 아래 노드가 활성화됩니다.

| 노드 | 설명 |
|------|------|
| Stable Fast 3D Loader | 모델 로드 |
| Stable Fast 3D Sampler | 이미지 → 메시 변환 |
| Stable Fast 3D Preview | 메시 미리보기 |
| Stable Fast 3D Save | GLB 파일 저장 |

---

## 디바이스 선택 규칙

| 환경 | 사용 디바이스 |
|------|--------------|
| CUDA 사용 가능 | `cuda` |
| Apple Silicon (MPS) | `mps` |
| 그 외 | `cpu` |
| `SF3D_USE_CPU=1` 환경 변수 설정 시 | 항상 `cpu` |

---

## 출력 파일

`output/<index>/mesh.glb` 형식으로 저장됩니다.  
GLB 파일은 Blender, Windows 3D 뷰어, Three.js 등에서 바로 열 수 있습니다.
