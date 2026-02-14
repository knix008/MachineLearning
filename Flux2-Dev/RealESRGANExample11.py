import logging
import warnings

# Suppress verbose logging from libraries before importing them
logging.getLogger("basicsr").setLevel(logging.WARNING)
logging.getLogger("realesrgan").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

import gradio as gr
import cv2
import numpy as np
import os
import sys
import signal
import atexit
import re
from PIL import Image
import time
import datetime
import gc
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


def cleanup():
    """Release all resources before exit."""
    print("Releasing resources...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Resources released!")


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle keyboard interrupt."""
    print("\nKeyboard interrupt received...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def print_progress_bar(ratio, desc, start_time, end=False):
    """커맨드라인에 프로그레스 바를 출력합니다."""
    elapsed = time.time() - start_time
    bar_len = 30
    filled = int(bar_len * ratio)
    bar = "█" * filled + "░" * (bar_len - filled)
    line = f"\r  [{bar}] {ratio*100:.0f}% - {desc} ({elapsed:.1f}초 경과)"
    sys.stdout.write(line)
    sys.stdout.flush()
    if end:
        sys.stdout.write("\n")


class ProgressCapture:
    """stdout/stderr 출력을 가로채서 타일 진행률을 Gradio progress와 콘솔에 반영합니다."""

    def __init__(self, original, progress_fn, start_time, start=0.3, end=0.8):
        self.original = original
        self.progress_fn = progress_fn
        self.start_time = start_time
        self.start = start
        self.end = end

    def write(self, text):
        # "X/Y" 패턴에서 진행률 추출 (예: "Tile 3/16", "Testing 5/20")
        match = re.search(r"(\d+)\s*/\s*(\d+)", text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0 and 0 < current <= total:
                ratio = current / total
                elapsed = time.time() - self.start_time
                val = self.start + ratio * (self.end - self.start)
                desc = f"타일 처리 중... ({current}/{total})"
                self.progress_fn(val, desc=f"{desc} - {elapsed:.1f}초 경과")
                # 콘솔 프로그레스 바 출력
                bar_len = 30
                filled = int(bar_len * val)
                bar = "█" * filled + "░" * (bar_len - filled)
                self.original.write(
                    f"\r  [{bar}] {val*100:.0f}% - {desc} ({elapsed:.1f}초 경과)"
                )
                self.original.flush()
                if current == total:
                    self.original.write("\n")
                return len(text)
        # 진행률 패턴이 아닌 출력은 무시 (라이브러리의 verbose 출력 억제)
        return len(text)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def get_model():
    """
    업스케일에 사용할 RRDBNet 모델과 관련 정보를 반환합니다.
    Returns:
        model: 업스케일용 네트워크 객체
        model_path: 가중치 파일 경로
        netscale: 업스케일 배수(4)
        dni_weight: DNI 가중치(혼합 비율, None 또는 리스트)
        device: torch.device 객체
    """
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    model_path = os.path.join("weights", "RealESRGAN_x4plus.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
    dni_weight = None
    # CUDA > MPS > CPU 순서로 자동 선택
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_str = "mps"
    else:
        device = torch.device("cpu")
        device_str = "cpu"
    print(f"[INFO] Using device: {device_str}")
    return model, model_path, dni_weight, device


def enhance_image(
    input_img,
    upscale,
    tile,
    tile_pad,
    pre_pad,
    fp32,
    ext,
    progress=gr.Progress(track_tqdm=True),
):
    """
    입력 이미지를 업스케일하고, 결과 이미지를 반환합니다.
    Args:
        input_img (PIL.Image): 업스케일할 이미지
        upscale (int): 업스케일 배수
        tile (int): 타일 크기 (0은 전체 처리)
        tile_pad (int): 타일 경계 패딩
        pre_pad (int): 전체 이미지 패딩
        fp32 (bool): FP32 연산 사용 여부
        ext (str): 저장 확장자
        progress: Gradio progress tracker
    Returns:
        PIL.Image, str: 업스케일된 이미지와 상태 메시지
    """
    start_time = time.time()

    progress(0.0, desc="이미지 전처리 중...")
    print_progress_bar(0.0, "이미지 전처리 중...", start_time)
    img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    progress(0.1, desc="모델 로딩 중...")
    print_progress_bar(0.1, "모델 로딩 중...", start_time)
    model, model_path, dni_weight, device = get_model()
    model = model.to(device)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=not fp32,
        gpu_id=None,
        device=device,
    )

    progress(0.3, desc="업스케일 처리 중...")
    print_progress_bar(0.3, "업스케일 처리 중...", start_time)
    capture_out = ProgressCapture(sys.stdout, progress, start_time, 0.3, 0.8)
    capture_err = ProgressCapture(sys.stderr, progress, start_time, 0.3, 0.8)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = capture_out
        sys.stderr = capture_err
        output, _ = upsampler.enhance(img, outscale=upscale)
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        del img, model, upsampler
        gc.collect()
        return None, f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    progress(0.8, desc="후처리 중...")
    print_progress_bar(0.8, "후처리 중...", start_time)
    del img, model, upsampler
    gc.collect()

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(output)

    progress(0.9, desc="저장 중...")
    print_progress_bar(0.9, "저장 중...", start_time)
    ext_to_use = ext
    filename = f"RealESRGAN_Example06_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{ext_to_use}"
    output.save(filename)

    elapsed = time.time() - start_time
    progress(1.0, desc="완료!")
    print_progress_bar(1.0, "완료!", start_time, end=True)
    print(f"처리 시간: {elapsed:.2f}초 | 저장: {filename}")
    return output, f"완료! 처리 시간: {elapsed:.2f}초"


with gr.Blocks() as demo:
    gr.Markdown("# Real-ESRGAN 업스케일러 (Gradio 데모)")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(
                label="입력 이미지 (업스케일할 원본)",
                type="pil",
                height=500,
                value="sample.png",
            )
            upscale = gr.Slider(
                minimum=1,
                maximum=4,
                value=4,
                step=1,
                label="업스케일 배수 (이미지를 몇 배로 확대할지)",
                info="이미지를 몇 배로 확대할지 설정합니다. (1~4)",
            )
            tile = gr.Slider(
                minimum=4,
                maximum=512,
                value=8,  # 기본값을 더 작게 : 16에서 변경
                step=4,
                label="Tile 크기 (메모리 부족시 분할 처리)",
                info="이미지를 분할 처리할 타일 크기입니다. 메모리 부족 시 값을 줄이세요. 0은 전체 처리.",
            )
            tile_pad = gr.Number(
                minimum=0,
                maximum=512,
                value=10,
                label="Tile padding (타일 경계 패딩)",
                info="타일 경계에 추가로 패딩을 줄 픽셀 수입니다.",
            )
            pre_pad = gr.Number(
                minimum=0,
                maximum=512,
                value=0,
                label="Pre padding (전체 이미지 패딩)",
                info="입력 이미지 전체에 추가로 패딩을 줄 픽셀 수입니다.",
            )
            fp32 = gr.Checkbox(
                label="FP32 모드 사용 (고정소수점 연산)",
                value=False,  # 기본값을 False로
                info="체크 시 FP32(고정소수점) 연산을 사용합니다. (메모리 여유가 많을 때 권장)",
            )
            ext = gr.Radio(
                choices=["png", "jpg"],
                value="jpg",
                label="저장 확장자 (결과 파일 형식)",
                info="결과 이미지를 저장할 파일 형식입니다.",
            )
            btn = gr.Button("업스케일 실행")
        with gr.Column():
            output_img = gr.Image(label="결과 이미지 (업스케일 결과)", height=800)
            status = gr.Textbox(label="상태 메시지")

    btn.click(
        enhance_image,
        inputs=[input_img, upscale, tile, tile_pad, pre_pad, fp32, ext],
        outputs=[output_img, status],
    )

if __name__ == "__main__":
    try:
        demo.launch(share=False, inbrowser=True)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    finally:
        cleanup()
        sys.exit(0)
