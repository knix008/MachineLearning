# 3D 웹 뷰어 (3D Web Viewer)

Node.js + Three.js 기반의 브라우저 3D 파일 뷰어입니다.

## 주요 기능

- **다양한 3D 파일 형식 지원**: OBJ, GLB, GLTF, FBX, STL
- **파일 업로드**: 클릭(파일 선택 다이얼로그) 또는 드래그 앤 드롭
- **인터랙티브 3D 뷰어**: 마우스로 회전 · 확대/축소 · 이동
- **뷰어 오버레이 UI**:
  - 좌측 상단: 현재 줌 배율 실시간 표시
  - 우측 상단: 조명 설정 패널 (환경광 · 주조명 · 보조조명 · 색상)
  - 좌측 하단: X/Y/Z 축 인디케이터 (카메라 회전에 따라 동적으로 갱신)
- **사이드바 컨트롤**:
  - 파일 정보 및 모델 통계 (버텍스 · 삼각형 · 경계 상자)
  - 모델 배율 조절 (0.1× ~ 5×)
  - 회전 속도 조절 · 자동 회전
  - 배경 색상 변경
  - 와이어프레임 모드 · 그리드 토글

## 요구사항

- **Node.js** v14 이상
- **브라우저**: Chrome 89+, Firefox 108+, Safari 16.4+, Edge 89+
  - ES Module `importmap` 지원 브라우저 필요

## 설치 및 실행

```powershell
# 의존성 설치
npm install

# 개발 서버 (자동 재시작)
npm run dev

# 일반 실행
npm start
```

브라우저에서 접속: `http://localhost:3000`

## 사용 방법

### 파일 업로드
- 사이드바 업로드 영역 **클릭** → 파일 선택 다이얼로그
- 또는 파일을 업로드 영역으로 **드래그 앤 드롭**

### 3D 모델 조작
| 동작 | 입력 |
|------|------|
| 회전 | 마우스 왼쪽 드래그 |
| 확대/축소 | 마우스 휠 |
| 이동 (패닝) | 마우스 오른쪽 드래그 |

### 뷰어 버튼
| 버튼 | 기능 |
|------|------|
| 리셋 | 카메라 초기 위치로 복귀 |
| 와이어프레임 | 와이어프레임 모드 토글 |
| 그리드 | 바닥 그리드 표시/숨김 |

## 프로젝트 구조

```
3DWebVIewer/
├── public/
│   ├── index.html      # 메인 페이지 (importmap 포함)
│   ├── styles.css      # 다크 테마 스타일
│   └── viewer.js       # Three.js 뷰어 (ES 모듈)
├── server.js           # Express 정적 파일 서버
├── package.json
└── README.md
```

## 기술 스택

| 분류 | 기술 |
|------|------|
| 서버 | Node.js, Express.js |
| 클라이언트 | HTML5, CSS3, JavaScript ES Modules |
| 3D 엔진 | Three.js v0.160.0 (CDN, ES Module importmap) |
| 로더 | OBJLoader · GLTFLoader · FBXLoader · STLLoader |
| 카메라 컨트롤 | OrbitControls |

## Three.js 주요 구성

- **메인 씬**: PerspectiveCamera + OrbitControls + Ambient/Directional Lights
- **축 인디케이터**: 별도 씬 + 카메라 → `setViewport` / `setScissor`로 좌측 하단 인셋 렌더링
- **파일 로딩**: FileReader API → 각 포맷별 Three.js 로더 `parse()` 호출

## 브라우저 호환성

| 브라우저 | 최소 버전 | 비고 |
|----------|----------|------|
| Chrome | 89 | importmap 지원 |
| Firefox | 108 | importmap 지원 |
| Safari | 16.4 | importmap 지원 |
| Edge | 89 | importmap 지원 |

## 라이선스

MIT License
