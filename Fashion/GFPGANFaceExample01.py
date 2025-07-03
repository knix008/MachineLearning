import gradio as gr
from PIL import Image
import time
import numpy as np

try:
    from gfpgan import GFPGANer
except ImportError:
    raise ImportError(
        "GFPGAN 라이브러리가 설치되어 있지 않습니다. "
        "pip install gfpgan 명령어로 설치하세요."
    )

# GFPGAN 모델 다운로드 및 로딩
# 모델 Pretrained 가중치 다운로드 경로 : https://github.com/TencentARC/GFPGAN?tab=readme-ov-file
model_path = './gfpgan/weights/GFPGANv1.3.pth'  # 경로에 모델이 없다면 GFPGAN이 자동으로 다운로드함
gfpganer = GFPGANer(
    model_path=model_path,
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

def restore_face(input_img):
    start = time.time()
    # PIL 이미지를 numpy로 변환
    img = input_img.convert("RGB")
    img_np = np.array(img)
    # 복원 수행
    cropped_faces, restored_img_np, restored_img = gfpganer.enhance(
        img_np, has_aligned=False, only_center_face=False, paste_back=True
    )
    elapsed = time.time() - start
    # numpy 이미지를 PIL로 변환
    restored_img_pil = Image.fromarray(restored_img)
    return restored_img_pil, f"처리 시간: {elapsed:.2f}초"

iface = gr.Interface(
    fn=restore_face,
    inputs=gr.Image(type="pil", label="입력 이미지 (얼굴 포함)"),
    outputs=[
        gr.Image(type="pil", label="복원된 이미지"),
        gr.Textbox(label="처리 시간")
    ],
    title="GFPGAN 얼굴 복원기",
    description="얼굴이 흐릿한 사진을 또렷하게 복원합니다. (GFPGAN v1.4)"
)

if __name__ == "__main__":
    iface.launch()