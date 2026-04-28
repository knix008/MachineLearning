"""
RealESRGAN_x4plus 모델 가중치 다운로드 스크립트
Usage: python download_RealESRGAN_x4plus.py
"""

import urllib.request
import sys
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODELS = [
    {
        "name": "RealESRGAN x4plus",
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth",
    },
]


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded / total_size * 100, 100)
        mb_done = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        bar = "#" * int(percent // 2) + "-" * (50 - int(percent // 2))
        print(f"\r  [{bar}] {percent:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
    else:
        print(f"\r  {downloaded / 1024 / 1024:.1f} MB 다운로드됨", end="", flush=True)


def download_weights(skip_existing=True):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"저장 경로: {WEIGHTS_DIR.resolve()}\n")

    total = len(MODELS)
    for i, model in enumerate(MODELS, 1):
        dest = WEIGHTS_DIR / model["filename"]
        print(f"[{i}/{total}] {model['name']}")
        print(f"  파일: {model['filename']}")

        if dest.exists() and skip_existing:
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  이미 존재함 ({size_mb:.1f} MB) — 건너뜀\n")
            continue

        print(f"  다운로드 중...")
        try:
            urllib.request.urlretrieve(model["url"], dest, reporthook=progress_hook)
            print()  # 줄바꿈
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  완료 ({size_mb:.1f} MB)\n")
        except Exception as e:
            print(f"\n  오류: {e}\n")
            if dest.exists():
                dest.unlink()  # 불완전한 파일 삭제

    print("=" * 60)
    print("다운로드 결과:")
    for model in MODELS:
        dest = WEIGHTS_DIR / model["filename"]
        if dest.exists():
            size_mb = dest.stat().st_size / 1024 / 1024
            print(f"  [OK] {model['filename']} ({size_mb:.1f} MB)")
        else:
            print(f"  [FAIL] {model['filename']} — 없음")


if __name__ == "__main__":
    skip = "--force" not in sys.argv
    if not skip:
        print("--force: 기존 파일도 다시 다운로드합니다.\n")
    download_weights(skip_existing=skip)
