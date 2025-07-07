import cv2
import numpy as np
import gradio as gr
import os

# --- 설정 ---
SR_MODEL_PATH = "EDSR_x4.pb"
# --- 설정 끝 ---

def enhance_image_quality_gradio(
    image: np.ndarray,
    method: str,
    sharpen_strength: float,
    gaussian_ksize: int,
    median_ksize: int
) -> np.ndarray:
    """
    이미지 화질 향상 함수
    """
    if image is None:
        return None

    processed_image = image.copy()

    try:
        if method == '선명화':
            k = sharpen_strength
            kernel = np.array([[-1, -1, -1],
                               [-1,  8 + k, -1],
                               [-1, -1, -1]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)

        elif method == '노이즈 제거 (가우시안)':
            ksize = int(gaussian_ksize)
            if ksize % 2 == 0:
                ksize += 1
            processed_image = cv2.GaussianBlur(processed_image, (ksize, ksize), 0)

        elif method == '노이즈 제거 (미디언)':
            ksize = int(median_ksize)
            if ksize % 2 == 0:
                ksize += 1
            processed_image = cv2.medianBlur(processed_image, ksize)

        elif method == '초해상도 (Super-Resolution)':
            # x4만 지원
            if not SR_MODEL_PATH or not os.path.exists(SR_MODEL_PATH):
                return image
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(SR_MODEL_PATH)
            model_name = os.path.basename(SR_MODEL_PATH).split('_')[0].lower()
            try:
                sr.setModel(model_name, 4)
                processed_image = sr.upsample(processed_image)
            except Exception:
                return image

        return processed_image

    except Exception:
        return image

with gr.Blocks() as iface:
    gr.Markdown("## 🌟 AI 이미지 화질 향상 도구 (x4 초해상도 고정) 🌟")

    with gr.Row():
        inp_img = gr.Image(type="numpy", label="여기에 이미지를 업로드하세요")
        out_img = gr.Image(type="numpy", label="향상된 이미지")

    method_radio = gr.Radio(
        ["선명화", "노이즈 제거 (가우시안)", "노이즈 제거 (미디언)", "초해상도 (Super-Resolution)"],
        label="화질 향상 방법 선택",
        value="선명화"
    )

    sharpen_slider = gr.Slider(1, 15, value=1, step=0.1, label="선명화 강도 (1~15)")
    gaussian_slider = gr.Slider(1, 31, value=5, step=2, label="가우시안 커널 크기 (홀수, 1~31)")
    median_slider = gr.Slider(1, 31, value=5, step=2, label="미디언 커널 크기 (홀수, 1~31)")

    def update_params(method):
        return (
            gr.update(visible=method=="선명화"),
            gr.update(visible=method=="노이즈 제거 (가우시안)"),
            gr.update(visible=method=="노이즈 제거 (미디언)")
        )

    method_radio.change(
        update_params,
        inputs=method_radio,
        outputs=[sharpen_slider, gaussian_slider, median_slider]
    )

    gr.Markdown("**초해상도(Super-Resolution)는 x4만 지원합니다.**<br>EDSR_x4.pb 모델 파일이 필요합니다.", elem_id="sr_notice")

    btn = gr.Button("화질 향상 실행")

    def process(
        image, method, sharpen_strength, gaussian_ksize, median_ksize
    ):
        return enhance_image_quality_gradio(
            image, method, sharpen_strength, gaussian_ksize, median_ksize
        )

    btn.click(
        process,
        inputs=[inp_img, method_radio, sharpen_slider, gaussian_slider, median_slider],
        outputs=out_img
    )

if __name__ == "__main__":
    iface.launch(share=False)