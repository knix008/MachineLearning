import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
print("Loading Model is Complete!!!")

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png"
)
pipe_prior_output = pipe_prior_redux(image)
images = pipe(
    guidance_scale=2.5,
    num_inference_steps=30, # 50
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_prior_output,
).images
images[0].save("flux-dev-redux.png")
