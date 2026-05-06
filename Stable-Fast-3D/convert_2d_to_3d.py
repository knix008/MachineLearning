"""
단일(또는 여러) 2D 이미지를 UV·텍스처가 있는 3D 메시(GLB)로 변환합니다.

Stable Fast 3D 공식 파이프라인(run.py)을 그대로 호출합니다.
모델 가중치: https://huggingface.co/stabilityai/stable-fast-3d (게이트, HF 로그인 필요)
코드 저장소: https://github.com/Stability-AI/stable-fast-3d

사용 예:
  python convert_2d_to_3d.py demo_files/examples/chair1.png --output-dir output/
  python convert_2d_to_3d.py --help
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    run_py = root / "run.py"
    if not run_py.is_file():
        print(
            "run.py를 찾을 수 없습니다. "
            "이 스크립트는 저장소 루트(https://github.com/Stability-AI/stable-fast-3d)에 두고 실행하세요.",
            file=sys.stderr,
        )
        sys.exit(1)
    cmd = [sys.executable, str(run_py), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
