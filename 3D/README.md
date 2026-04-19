# 2D 이미지 → 깊이 기반 3D 표현 (`Convert-2d-to-3d-example01.py`)

단일 RGB 이미지에서 **[Intel DPT (dpt-hybrid-midas)](https://huggingface.co/Intel/dpt-hybrid-midas)** 로 깊이를 추정한 뒤, **깊이 맵 이미지**, **색 점군(.ply)**, **높이장 메시(.obj)** 를 만듭니다.  
한 장의 사진으로 물체 뒤까지 복원하는 진짜 볼륨 3D가 아니라, 보이는 면을 깊이로 밀어 올린 **2.5D(완화된 3D)** 에 가깝습니다.

## 실행 장치 (자동 감지)

별도 설정 없이 PyTorch가 하드웨어를 판별합니다.

| 우선순위 | 조건 |
|----------|------|
| 1. **CUDA** | `torch.cuda.is_available()` 가 참일 때 (일반적으로 NVIDIA GPU) |
| 2. **MPS** | CUDA가 없고 Apple **MPS** 가 사용 가능할 때 (Apple Silicon 등) |
| 3. **CPU** | 위 가속기가 없을 때 |

- **CLI**: 변환 직전에 한 줄 요약이 출력되고, 깊이 모델을 처음 불러올 때 콘솔에 `torch.device` 로그가 나갑니다.
- **Gradio**: 화면 상단 **「실행 장치 (자동 감지: CUDA → MPS → CPU)」** 칸에 현재 감지 결과가 표시됩니다.

GPU가 있는데도 CPU로만 동작하면, 설치된 `torch`가 CPU 전용 빌드이거나 드라이버/CUDA 조합이 맞지 않은 경우가 많습니다. 확인:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

자세한 설치·CUDA 휠 안내는 이 폴더의 **`requirements.txt`** 상단 주석을 참고하세요.

## 지원 입력 형식

| 형식 | 비고 |
|------|------|
| `.jpg` / `.jpeg` | |
| `.png` | |
| `.gif` | **첫 프레임**만 사용 |
| `.webp` | 다중 프레임인 경우 **첫 프레임**만 사용 |

## 출력

- `*_depth.png` — 깊이를 의사 컬러로 시각화한 2D 이미지  
- `*_points.ply` — RGB가 붙은 ASCII 점군 (MeshLab 등에서 열기)  
- `*_mesh.obj` — 깊이 그리드 기반 삼각형 메시 (색은 PLY 쪽 권장)

기본 저장 위치: 이 폴더 아래 `image_to_3d_out/` (`.gitignore`에 포함됨)

## 설치

```bash
cd 3D
pip install -r requirements.txt
```

NVIDIA GPU를 쓰려면 [PyTorch 공식 안내](https://pytorch.org/get-started/locally/)에 맞는 **CUDA 포함 `torch`** 를 설치한 뒤, 위처럼 `cuda: True` 인지 확인하는 것이 좋습니다.

최초 실행 시 Hugging Face에서 가중치(`Intel/dpt-hybrid-midas`)가 내려받아집니다.

## CLI

```bash
python Convert-2d-to-3d-example01.py path/to/photo.png -o image_to_3d_out
```

주요 옵션:

- `-o`, `--output-dir` — 출력 디렉터리 (기본: `image_to_3d_out`)
- `--sample-step` — 점군 샘플 간격 (클수록 가벼움, 기본: 2)
- `--mesh-max-side` — OBJ 그리드용 최대 변 길이 픽셀 (기본: 256)
- `--z-scale` — 깊이 돌출 강도 (기본: 2.0)

## Gradio UI

```bash
python Convert-2d-to-3d-example01.py --gradio
```

- **입력:** 파일 선택기로 **JPG / JPEG / PNG / GIF / WebP** 만 선택 가능합니다.  
- **출력:** 입력 RGB 미리보기, **깊이 맵** 이미지, **OBJ**용 **Gradio `Model3D`**(브라우저에서 회전·확대, Gradio 4.x 권장), **PLY / OBJ** 다운로드, 저장 경로 로그.

## 이 폴더의 다른 파일

- `3DObjectDisplay01.py` — 업로드한 `.obj` / `.stl` / `.ply` 등을 Plotly로 보는 뷰어 (별도 의존성: `plotly`, `trimesh` 등)
- `requirements.txt` — 패키지 목록 및 PyTorch/GPU 설치 안내

## 라이선스·크레딧

- 사용 모델의 라이선스는 [Hugging Face 모델 카드](https://huggingface.co/Intel/dpt-hybrid-midas)를 따릅니다.
