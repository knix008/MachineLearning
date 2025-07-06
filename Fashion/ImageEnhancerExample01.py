import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def enhance_image(input_img, denoise_strength, sharpen_strength, contrast, brightness):
    # PIL 이미지를 OpenCV로 변환
    img = np.array(input_img.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 노이즈 제거 (denoise_strength가 0이면 생략)
    if denoise_strength > 0:
        # h, hColor 값에 따라 노이즈 제거 강도 조절
        img = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=denoise_strength,
            hColor=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # 선명화 (sharpen_strength가 0이면 생략)
    if sharpen_strength > 0:
        # 샤프닝 커널 생성 (강도 조절)
        kernel = np.array([[0, -1, 0], [-1, 5 + sharpen_strength, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

    # OpenCV 이미지를 PIL로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 대비 조절 (1.0=원본)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(contrast)

    # 밝기 조절 (1.0=원본)
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(brightness)

    return img_pil


with gr.Blocks(title="이미지 품질 향상") as demo:
    gr.Markdown("## 이미지 노이즈 제거 및 선명화/대비/밝기 조절 도구")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="원본 이미지", type="pil")
            denoise_slider = gr.Slider(
                0, 30, value=10, step=1, label="노이즈 제거 강도 (0=적용 안함)"
            )
            sharpen_slider = gr.Slider(
                0, 10, value=2, step=1, label="선명화 강도 (0=적용 안함)"
            )
            contrast_slider = gr.Slider(
                0.5, 2.0, value=1.2, step=0.05, label="대비 (1.0=원본)"
            )
            brightness_slider = gr.Slider(
                0.5, 2.0, value=1.0, step=0.05, label="밝기 (1.0=원본)"
            )
            btn = gr.Button("이미지 향상")
        with gr.Column():
            out = gr.Image(label="향상된 이미지", type="pil")
    btn.click(
        enhance_image,
        inputs=[
            inp,
            denoise_slider,
            sharpen_slider,
            contrast_slider,
            brightness_slider,
        ],
        outputs=out,
    )

if __name__ == "__main__":
    demo.launch()
