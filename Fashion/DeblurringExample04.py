import gradio as gr
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
import numpy as np


def deblur_and_enhance(image, half_precision=True, scale=4):
    """
    업로드된 이미지를 Real-ESRGAN 딥러닝 모델로 더욱 효과적으로 선명하게 복원합니다.
    - scale: 업스케일 배수 (권장: 2 또는 4)
    """
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path="RealESRGAN_x4plus.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
    )

    img = image.convert("RGB")
    img_np = np.array(img)

    try:
        output, _ = upsampler.enhance(img_np, outscale=scale)
    except RuntimeError as error:
        return (
            None,
            f"에러 발생: {error}\nNVIDIA 드라이버가 설치된 GPU가 필요할 수 있습니다. CPU로 실행하려면 half precision 옵션을 꺼주세요.",
        )
    result_img = Image.fromarray(output)
    return result_img, "복원 및 품질 향상 완료!"


with gr.Blocks() as demo:
    gr.Markdown(
        "## 딥러닝 기반 이미지 선명화 및 품질 향상 (Real-ESRGAN)\n이미지를 업로드하면 딥러닝으로 더욱 선명하게 복원하고 업스케일합니다."
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="입력 이미지", type="pil")
            half_precision = gr.Checkbox(label="half precision (GPU 권장)", value=True)
            scale = gr.Slider(
                label="업스케일 배수 (2~4)", minimum=2, maximum=4, value=4, step=1
            )
            submit_btn = gr.Button("이미지 복원 및 향상")
        with gr.Column():
            output_image = gr.Image(label="복원된 이미지")
            status_text = gr.Textbox(label="상태 메시지")

    submit_btn.click(
        fn=deblur_and_enhance,
        inputs=[input_image, half_precision, scale],
        outputs=[output_image, status_text],
    )

if __name__ == "__main__":
    demo.launch()
