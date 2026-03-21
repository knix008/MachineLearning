import torch
from diffusers.utils import load_image

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel


base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)

# Enable multiple memory optimizations
pipe.enable_model_cpu_offload()  # Offload model to CPU when not in use
pipe.enable_sequential_cpu_offload()  # More aggressive CPU offloading
pipe.enable_attention_slicing(1)  # Slice attention computation
print("모델 로딩 완료!")


prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'
control_image_depth = load_image("https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/depth.jpg")
control_mode_depth = 2

control_image_canny = load_image("https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/canny.jpg")
control_mode_canny = 0

width, height = control_image_depth.size

image = pipe(
    prompt, 
    control_image=[control_image_depth, control_image_canny],
    control_mode=[control_mode_depth, control_mode_canny],
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.2, 0.4],
    num_inference_steps=24, 
    guidance_scale=3.5,
    generator=torch.manual_seed(42),
).images[0]

image.save("flux1_dev_controlnet01.png")