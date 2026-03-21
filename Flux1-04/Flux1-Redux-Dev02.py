import torch
import gradio as gr
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from PIL import Image
import datetime


pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev",
    torch_dtype=torch.bfloat16,
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    text_encoder=None,
    text_encoder_2=None,
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
print("Loading Model is Complete!!!")


def flux1_redux_run(
    input_image: Image.Image, guidance_scale=2.5, num_inference_steps=30, seed=0
):
    # Prior output 생성
    pipe_prior_output = pipe_prior_redux(input_image)
    # Generator 설정
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator("cpu").manual_seed(seed)

    # 이미지 생성
    images = pipe(
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        width=input_image.width,
        height=input_image.height,
        **pipe_prior_output,
    ).images
    result = images[0]
    filename = (
        f"Flux1-Redux-Dev02_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
    result.save(filename)
    param_info = f"""
    **Guidance Scale:** {guidance_scale}\n
    **Num Inference Steps:** {num_inference_steps}\n
    **Seed:** {seed}\n
    **Input Image Size:** {input_image.width}x{input_image.height}
    **Output Image Size:** {result.width}x{result.height}
    """
    return result, param_info


with gr.Blocks() as demo:
    gr.Markdown("# Flux1-Redux Pipeline Example")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input Image", type="pil", value="default.jpg", height=500
            )
            guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=1.0, maximum=10.0, value=2.5, step=0.1
            )
            num_inference_steps = gr.Slider(
                label="Num Inference Steps", minimum=1, maximum=50, value=30, step=1
            )
            seed = gr.Number(label="Seed", value=-1, precision=0)
            run_btn = gr.Button("Run")
        with gr.Column():
            output_img = gr.Image(label="Output Image", height=500)
            param_info_md = gr.Markdown(label="Parameter Info")

    def run_model(input_image, guidance_scale, num_inference_steps, seed):
        if input_image is None:
            return None, "이미지를 업로드하세요."
        return flux1_redux_run(input_image, guidance_scale, num_inference_steps, seed)

    run_btn.click(
        run_model,
        inputs=[input_image, guidance_scale, num_inference_steps, seed],
        outputs=[output_img, param_info_md],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
