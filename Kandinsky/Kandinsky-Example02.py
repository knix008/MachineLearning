import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import datetime

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)

init_image.save("kandinsky-Example02-init.png")

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"


def my_callback(pipe, step, timestep, callback_kwargs):
    print(f"Step {step}, Timestep {timestep}")
    return {}  # Always return a dict, even if empty


image = pipeline(prompt, image=init_image, callback_on_step_end=my_callback).images[0]
make_image_grid([init_image, image], rows=1, cols=2)

image.save(
    f"kandinsky-Example02-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
)
