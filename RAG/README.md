# RAG 서비스 (RAGFlow)

이 디렉터리는 [RAGFlow](https://github.com/infiniflow/ragflow)(infiniflow) 기반 **자체 호스팅 RAG**를 구축하기 위한 래퍼입니다. 실제 스택은 공식 저장소의 `docker/` 구성을 그대로 사용합니다.

## 구성

| 항목 | 설명 |
|------|------|
| `scripts/bootstrap.ps1` / `bootstrap.sh` | `RAG/ragflow/`에 공식 저장소 클론. 선택적 **태그/브랜치** 인자 지원 |
| `scripts/up.ps1` / `up.sh` | CPU 기본(`DEVICE`는 upstream `.env` 기준)으로 기동 |
| `scripts/up-gpu.ps1` / `up-gpu.sh` | 호스트 환경변수 `DEVICE=gpu`로 기동(DeepDoc GPU; NVIDIA Container Toolkit 필요) |
| `scripts/down.ps1` / `down.sh` | 스택 중지 |
| `scripts/apply-overrides.ps1` / `apply-overrides.sh` | `config/env.override`의 키를 `ragflow/docker/.env`에 반영(백업 `.env.bak`) |
| `scripts/init-submodule.ps1` / `init-submodule.sh` | 저장소 루트에서 `RAG/ragflow`를 **git submodule**로 등록 |
| `config/env.override.example` | → `config/env.override`로 복사 후 `apply-overrides` 실행 |

`ragflow/`는 기본적으로 `.gitignore`에 있어 **클론만** 쓰는 방식에 맞춰 두었습니다. submodule을 쓰면 **`.gitignore`에서 `ragflow/` 한 줄을 제거**해야 합니다.

## 사전 요구 사항

- **Docker** ≥ 24, **Docker Compose** ≥ v2.26 ([Install Docker Engine](https://docs.docker.com/engine/install/))
- **CPU** ≥ 4코어, **RAM** ≥ 16GB, **디스크** ≥ 50GB 권장 (공식 README 기준)
- 이미지는 **linux/amd64** 위주; ARM64는 공식 Docker 이미지가 제한적일 수 있음
- **Linux / WSL2**: Elasticsearch용으로 `vm.max_map_count` ≥ **262144** 필요

  ```bash
  sudo sysctl -w vm.max_map_count=262144
  # 영구 적용: /etc/sysctl.conf 에 vm.max_map_count=262144 추가 후 sudo sysctl -p
  ```

- **GPU**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치 후 `up-gpu` 사용. 필요 시 `config/env.override`에 `DEVICE=gpu`를 넣고 `apply-overrides`로 `docker/.env`에 고정해도 됩니다.

## 빠른 시작 (Windows PowerShell)

```powershell
cd RAG
.\scripts\bootstrap.ps1
# 선택: 버전 고정
# .\scripts\bootstrap.ps1 -Tag v0.24.0
# 선택: 로컬 덮어쓰기 (GPU·이미지 핀 등)
# copy config\env.override.example config\env.override
# notepad config\env.override
# .\scripts\apply-overrides.ps1
.\scripts\up.ps1
# 또는 GPU:
# .\scripts\up-gpu.ps1
```

로그 확인(컨테이너 이름은 환경에 따라 다를 수 있음):

```powershell
docker logs -f docker-ragflow-cpu-1
```

브라우저: 기본은 **`http://localhost`**. API 포트는 `ragflow/docker/.env`의 `SVR_HTTP_PORT` 등을 참고.

## 호스트의 Ollama와 RAGFlow 연결 (Docker)

RAGFlow는 **컨테이너** 안에서 동작합니다. 설정에 **`http://127.0.0.1:11434`** 또는 **`http://localhost:11434`**를 넣으면, 그 `localhost`는 **PC가 아니라 컨테이너 자신**이라 Ollama에 닿지 않습니다.

- **Docker Desktop (Windows/Mac)**  
  베이스 URL을 **`http://host.docker.internal:11434`** 로 지정하세요. (API 키는 로컬 Ollama에 보통 불필요)

- **Linux 네이티브 Docker**  
  `host.docker.internal`이 없을 수 있으면 **호스트 LAN IP**(예: `192.168.x.x:11434`) 또는 [추가 호스트 설정](https://docs.docker.com/desktop/features/networking/#i-want-to-connect-from-a-container-to-a-service-on-the-host)을 사용하세요.

확인 예시(컨테이너에서 호스트 Ollama로 태그 목록 조회):

```powershell
docker exec docker-ragflow-cpu-1 curl -sS http://host.docker.internal:11434/api/tags
```

## 빠른 시작 (Linux / macOS)

```bash
cd RAG
chmod +x scripts/*.sh
./scripts/bootstrap.sh
# ./scripts/bootstrap.sh v0.24.0
# cp config/env.override.example config/env.override && ${EDITOR:-vi} config/env.override
# ./scripts/apply-overrides.sh
./scripts/up.sh
# 또는: ./scripts/up-gpu.sh
```

## Git submodule으로 고정

1. **`RAG/ragflow`가 없을 때** (bootstrap으로 만든 폴더가 있으면 삭제)
2. 저장소 루트에서:

   ```powershell
   .\RAG\scripts\init-submodule.ps1
   ```

   또는 Linux:

   ```bash
   ./RAG/scripts/init-submodule.sh
   ```

3. **`RAG/.gitignore`에서 `ragflow/` 줄 삭제** 후 `.gitmodules`와 함께 커밋.

## 설정을 바꿀 때

- **`ragflow/docker/.env`**: 포트, `RAGFLOW_IMAGE`, `DEVICE`, DB/MinIO 비밀번호 등
- **`config/env.override` + `apply-overrides`**: 이 레포에만 둘 커스텀을 분리해 관리 (실제 비밀값은 커밋하지 말 것; `.gitignore`에 `env.override` 포함)
- **`ragflow/docker/service_conf.yaml.template`**: 기본 LLM 팩토리, API 키 템플릿 등
- 변경 후: `ragflow/docker`에서 `docker compose -f docker-compose.yml up -d` 로 재기동 (또는 `up.ps1` / `up-gpu.ps1`)

## 참고 문서

- [RAGFlow README (Get Started)](https://github.com/infiniflow/ragflow/blob/main/README.md)
- [docker/README.md](https://github.com/infiniflow/ragflow/blob/main/docker/README.md)
- [RAGFlow 문서](https://ragflow.io/docs/dev/)
