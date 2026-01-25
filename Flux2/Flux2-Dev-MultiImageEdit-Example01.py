import torch
from diffusers import Flux2Pipeline, Flux2PriorReduxPipeline, Flux2Img2ImgPipeline
from datetime import datetime
from PIL import Image
import os
import warnings
import gradio as gr
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", message=".*add_prefix_spade.*")
warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
warnings.filterwarnings("ignore", message=".*slow tokenizers.*")

# Set device and data type
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Device: {device}, dtype: {dtype}")
print("모델 로딩 중...")

# Load FLUX.2 Redux pipeline for multi-image input
pipe_redux = Flux2PriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-Redux-dev", torch_dtype=dtype
).to(device)

# Load base FLUX.2 pipeline
pipe_base = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=dtype
).to(device)

# Load Img2Img pipeline for image-to-image generation
pipe_img2img = Flux2Img2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=dtype
).to(device)

# Enable memory optimizations
if device == "cpu":
    pipe_base.enable_model_cpu_offload()
    pipe_base.enable_attention_slicing(1)
    pipe_img2img.enable_model_cpu_offload()
    pipe_img2img.enable_attention_slicing(1)
else:
    pipe_base.enable_model_cpu_offload()
    pipe_img2img.enable_model_cpu_offload()

print("모델 로딩 완료!")


def resize_image_to_multiple_of_64(image, target_width=None, target_height=None):
    """
    Resize image so dimensions are multiples of 64
    
    Parameters:
    -----------
    image : PIL.Image
        Input image
    target_width : int
        Target width (if None, use image width)
    target_height : int
        Target height (if None, use image height)
    
    Returns:
    --------
    PIL.Image
        Resized image
    """
    width = target_width or image.width
    height = target_height or image.height
    
    # Round to nearest multiple of 64
    width = (width // 64) * 64
    height = (height // 64) * 64
    
    # Ensure minimum size
    width = max(width, 256)
    height = max(height, 256)
    
    return image.resize((width, height), Image.Resampling.LANCZOS)


def generate_from_multi_images(
    image1, image2, image3, image4,
    width, height, guidance_scale, num_inference_steps, seed,
    mode, strength
):
    """
    여러 이미지를 입력받아 하나의 이미지를 생성합니다.

    Parameters:
    -----------
    image1-4 : PIL.Image
        입력 이미지들 (최소 1개 필요)
    width, height : int
        출력 이미지 크기
    guidance_scale : float
        프롬프트 강도
    num_inference_steps : int
        추론 스텝 수
    seed : int
        랜덤 시드
    mode : str
        생성 모드 ("redux" 또는 "img2img")
    strength : float
        이미지 변환 강도
    """
    try:
        # Collect valid images
        images = [img for img in [image1, image2, image3, image4] if img is not None]

        if len(images) == 0:
            return None, "✗ 오류: 최소 1개의 이미지를 제공해주세요"

        print(f"입력 이미지 수: {len(images)}")

        # Ensure dimensions are multiples of 64
        width = int(max((width // 64) * 64, 256))
        height = int(max((height // 64) * 64, 256))

        print(f"생성 설정: {width}x{height}, guidance={guidance_scale}, steps={num_inference_steps}")

        generator = torch.Generator(device=device).manual_seed(int(seed))

        if mode == "redux":
            # FLUX Redux 모드: 여러 이미지의 특징을 결합
            print("Redux 모드: 이미지 특징 결합 중...")

            # Process images through Redux pipeline
            # Redux can accept multiple images and blend their features
            redux_output = pipe_redux(images)

            # Generate final image using the combined embeddings
            image = pipe_base(
                prompt_embeds=redux_output.prompt_embeds,
                pooled_prompt_embeds=redux_output.pooled_prompt_embeds,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                generator=generator,
            ).images[0]

        else:
            # Img2Img 모드: 첫 번째 이미지를 기반으로 변환
            print("Img2Img 모드: 이미지 변환 중...")

            # Use first image as base
            base_image = images[0].convert("RGB")
            base_image = resize_image_to_multiple_of_64(base_image, width, height)

            # If multiple images, blend them together as base
            if len(images) > 1:
                print(f"{len(images)}개 이미지 블렌딩 중...")
                blended = blend_images(images, width, height)
                base_image = blended

            # Get prompt from Redux if available
            redux_output = pipe_redux(images[0])

            image = pipe_img2img(
                image=base_image,
                prompt_embeds=redux_output.prompt_embeds,
                pooled_prompt_embeds=redux_output.pooled_prompt_embeds,
                width=width,
                height=height,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=int(num_inference_steps),
                generator=generator,
            ).images[0]

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}_{timestamp}.png"
        image.save(filename)

        return image, f"✓ 이미지가 저장되었습니다: {filename}"

    except Exception as e:
        import traceback
        error_msg = f"✗ 오류 발생: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg


def blend_images(images, width, height):
    """여러 이미지를 블렌딩하여 하나의 이미지로 만듭니다."""
    # Resize all images to target size
    resized = []
    for img in images:
        img_rgb = img.convert("RGB")
        img_resized = img_rgb.resize((width, height), Image.Resampling.LANCZOS)
        resized.append(np.array(img_resized, dtype=np.float32))

    # Average blend
    blended = np.mean(resized, axis=0).astype(np.uint8)
    return Image.fromarray(blended)


# Create Gradio interface
with gr.Blocks(title="Flux.1-dev Multi-Image Editor") as interface:
    gr.Markdown("# 🎨 Flux.1-dev 멀티 이미지 에디터")
    gr.Markdown("여러 이미지를 입력받아 FLUX Redux를 사용하여 하나의 이미지를 생성합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            # Input images - 4개까지 지원
            gr.Markdown("### 입력 이미지 (최대 4개)")
            with gr.Row():
                image1 = gr.Image(
                    label="이미지 1 (필수)",
                    type="pil",
                    sources=["upload"],
                    height=200,
                )
                image2 = gr.Image(
                    label="이미지 2 (선택)",
                    type="pil",
                    sources=["upload"],
                    height=200,
                )
            with gr.Row():
                image3 = gr.Image(
                    label="이미지 3 (선택)",
                    type="pil",
                    sources=["upload"],
                    height=200,
                )
                image4 = gr.Image(
                    label="이미지 4 (선택)",
                    type="pil",
                    sources=["upload"],
                    height=200,
                )

            # Generation mode
            mode = gr.Radio(
                label="생성 모드",
                choices=["redux", "img2img"],
                value="redux",
                info="redux: 여러 이미지 특징 결합 / img2img: 첫 이미지 기반 변환",
            )

            with gr.Row():
                width = gr.Number(
                    label="이미지 너비",
                    value=512,
                    step=64,
                    precision=0,
                    info="생성할 이미지의 너비 (픽셀). 64의 배수.",
                )
                height = gr.Number(
                    label="이미지 높이",
                    value=768,
                    step=64,
                    precision=0,
                    info="생성할 이미지의 높이 (픽셀). 64의 배수.",
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=4.0,
                    info="이미지 특징을 얼마나 강하게 따를지 제어합니다.",
                )
                num_inference_steps = gr.Slider(
                    label="추론 스텝",
                    minimum=10,
                    maximum=50,
                    step=1,
                    value=28,
                    info="높을수록 품질이 좋지만 느립니다.",
                )

            with gr.Row():
                seed = gr.Number(
                    label="시드",
                    value=100,
                    precision=0,
                    info="같은 시드 = 같은 결과",
                )
                strength = gr.Slider(
                    label="강도 (img2img 모드용)",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=0.8,
                    info="img2img 모드에서 원본 이미지 변환 강도",
                )

            generate_btn = gr.Button("🚀 이미지 생성", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(label="생성된 이미지", height=600)
            output_message = gr.Textbox(label="상태", interactive=False)

    # Connect the generate button to the function
    generate_btn.click(
        fn=generate_from_multi_images,
        inputs=[
            image1, image2, image3, image4,
            width, height,
            guidance_scale, num_inference_steps, seed,
            mode, strength,
        ],
        outputs=[output_image, output_message],
    )

    gr.Markdown("---")
    gr.Markdown(
        """
    ### 사용 방법:

    **Redux 모드** (권장)
    - 여러 이미지의 스타일과 특징을 결합하여 새로운 이미지를 생성합니다
    - 예: 인물 사진 + 배경 사진 → 결합된 이미지
    - 최대 4개 이미지의 특징을 블렌딩합니다

    **Img2Img 모드**
    - 첫 번째 이미지를 기반으로 변환합니다
    - 여러 이미지가 있으면 평균 블렌딩 후 변환
    - strength 값으로 변환 강도 조절

    ### 파라미터 설명:

    **입력 이미지**
    - 최소 1개, 최대 4개의 이미지를 업로드할 수 있습니다
    - 이미지 1은 필수입니다

    **Guidance Scale**
    - 입력 이미지 특징을 얼마나 강하게 따를지 제어합니다
    - 권장값: 3-7

    **추론 스텝**
    - 이미지 생성 단계 수입니다
    - 권장값: 20-28

    **시드** (Seed)
    - 난수 생성의 시작점입니다
    - 같은 시드를 사용하면 같은 결과를 얻습니다

    **강도** (Strength)
    - img2img 모드에서 원본 이미지를 얼마나 변환할지 제어
    - 낮을수록 원본 유지, 높을수록 많이 변환
    """
    )

# Launch the interface
if __name__ == "__main__":
    interface.launch(inbrowser=True)
