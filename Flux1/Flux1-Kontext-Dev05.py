import torch
import gradio as gr
import time
from diffusers import FluxKontextPipeline
from PIL import Image
import datetime

# Load the pipeline (make sure the model is downloaded or available locally)
pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev")
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
print("Loading Model is Complete!!!")

# Define maximum image size
MAX_IMAGE_SIZE = 1024

def resize_image_keep_ratio(image: Image.Image, max_size: int) -> Image.Image:
    """
    이미지의 가로, 세로 중 큰 쪽이 max_size를 넘지 않도록 비율을 유지하며 리사이즈합니다.
    리사이즈 후 가로, 세로는 16의 배수로 맞춥니다.
    """
    if image is None:
        return None
    w, h = image.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # 16의 배수로 내림
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    # 최소 크기 보장
    new_w = max(new_w, 16)
    new_h = max(new_h, 16)
    img = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"The image size : {img.width}, {img.height}")
    return img

def flux1_kontext_dev(
    prompt,
    input_image: Image.Image,
    guidance=2.5,
    num_inference_steps=30,
    seed=-1,
):
    # 입력 이미지 리사이즈 (비율 유지, 최대 크기 제한)
    resized_image = resize_image_keep_ratio(input_image, MAX_IMAGE_SIZE)

    # Seed generator
    generator = None
    if seed is not None and str(seed).strip() != "" and int(seed) != -1:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    img_width = resized_image.width
    img_height = resized_image.height

    # Run the pipeline
    result = pipe(
        prompt=prompt,
        image=resized_image,
        width=img_width,
        height=img_height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    result.save(
        f"Flux1-Kontext-Dev05_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    # 파라미터 정보 문자열 생성
    param_info = f"""
    **Prompt:** {prompt}\n
    **Guidance Scale:** {guidance}\n
    **Num Inference Steps:** {num_inference_steps}\n
    **Seed:** {seed if seed is not None and str(seed).strip() != '' else '-1 (random)'}\n
    **Image Size:** {img_width}x{img_height}
    """
    return result, param_info


with gr.Blocks() as demo:
    gr.Markdown("# Flux1-Kontext-Dev")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                value="change her face to see the camera",
            )
            input_image = gr.Image(
                label="Input Image", value="default.jpg", type="pil", height=500
            )
            guidance = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=10.0, value=7.5, step=0.1
            )
            num_inference_steps = gr.Slider(
                label="Num Inference Steps", minimum=1, maximum=50, value=30, step=1
            )
            run_btn = gr.Button("Run")
        with gr.Column():
            output_img = gr.Image(label="Output Image", height=500)
            param_info_md = gr.Markdown(label="Parameter Info")

    seed = gr.Textbox(
        label="Seed (default: -1, random)", value="42", placeholder="-1 for random"
    )

    def run_model(
        prompt, input_image, guidance, num_inference_steps, seed
    ):
        output_img, param_info = flux1_kontext_dev(
            prompt, input_image, guidance, num_inference_steps, seed
        )
        return output_img, param_info

    run_btn.click(
        run_model,
        inputs=[
            prompt,
            input_image,
            guidance,
            num_inference_steps,
            seed,
        ],
        outputs=[output_img, param_info_md],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
