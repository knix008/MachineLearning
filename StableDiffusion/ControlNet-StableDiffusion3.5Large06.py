import torch
import gradio as gr
import cv2
import numpy as np
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import torchvision.transforms.functional as F


class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)

    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5

        # assuming img is a PIL image
        img = F.to_tensor(image)
        img = cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        img = cv2.Canny(img, 100, 200)
        
        return image

    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image


# 모델 로드
controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    text_encoder_device_map="auto",
    text_encoder_2_device_map="auto", 
    text_encoder_3_device_map="auto"
).to("cuda")

# Text encoder들을 명시적으로 CUDA로 이동
if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
    pipe.text_encoder = pipe.text_encoder.to("cuda")
if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
    pipe.text_encoder_3 = pipe.text_encoder_3.to("cuda")

pipe.image_processor = SD3CannyImageProcessor()


def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """입력 이미지를 Canny edge detection으로 전처리"""
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, low_threshold, high_threshold)
    canny_image = Image.fromarray(canny)
    return canny_image


def generate_image(input_image, prompt, negative_prompt, controlnet_conditioning_scale, guidance_scale, num_inference_steps, seed):
    """이미지 생성 함수"""
    if input_image is None:
        return None, "입력 이미지를 선택해주세요."
    
    try:
        # Canny edge detection 적용
        control_image = preprocess_canny(input_image)
        
        # 시드 설정
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # 이미지 생성
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            max_sequence_length=77,
        )
        
        generated_image = result.images[0]
        return generated_image, "이미지 생성 완료!"
        
    except Exception as e:
        return None, f"오류 발생: {str(e)}"


# Gradio 인터페이스 구성
with gr.Blocks(title="Stable Diffusion 3.5 ControlNet Canny") as demo:
    gr.Markdown("# Stable Diffusion 3.5 Large ControlNet (Canny)")
    gr.Markdown("입력 이미지의 윤곽선을 기반으로 새로운 이미지를 생성합니다.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="입력 이미지",
                type="pil",
                height=400,
                value="default_image.jpg",  # 기본 이미지 경로
            )
            prompt = gr.Textbox(
                label="프롬프트",
                placeholder="생성할 이미지에 대한 설명을 입력하세요...",
                lines=3,
                value="A beautiful woman in a red bikini, standing on a beach at sunset, with a serene expression"
            )
            negative_prompt = gr.Textbox(
                label="네거티브 프롬프트",
                placeholder="생성하지 않을 요소들을 입력하세요 (예: blurry, low quality, deformed)...",
                lines=2,
                value="blurry, low quality, deformed, bad anatomy, extra limbs, poorly drawn hands"
            )
            
            with gr.Row():
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="ControlNet Strength",
                    info="ControlNet의 강도를 조절합니다. 0은 ControlNet을 사용하지 않음을 의미합니다."
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale",
                    info="높은 값은 더 창의적인 결과를 생성하지만, 너무 높으면 품질이 저하될 수 있습니다."
                )
            
            with gr.Row():
                num_inference_steps = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=60,
                    step=10,
                    label="Inference Steps",
                    info="더 많은 단계는 더 나은 품질을 제공하지만 시간이 더 걸립니다"
                )
                seed = gr.Number(
                    label="시드",
                    value=0,
                    precision=0,
                    info="이미지 생성의 무작위성을 제어합니다. 동일한 시드를 사용하면 동일한 결과를 얻을 수 있습니다."
                )
            
            generate_btn = gr.Button("이미지 생성", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(
                label="생성된 이미지",
                height=400
            )
            status_text = gr.Textbox(
                label="상태",
                interactive=False
            )
    
    # 이벤트 바인딩
    generate_btn.click(
        fn=generate_image,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            controlnet_conditioning_scale,
            guidance_scale,
            num_inference_steps,
            seed
        ],
        outputs=[output_image, status_text]
    )


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
