import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor


class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)

    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image

    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image


controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")
pipe.image_processor = SD3CannyImageProcessor()

control_image = load_image(
    "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/canny.png"
)
prompt = "A Night time photo taken by Leica M11, portrait of a Japanese woman in a kimono, looking at the camera, Cherry blossoms"

generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=1.0,
    guidance_scale=3.5,
    num_inference_steps=60,
    generator=generator,
    max_sequence_length=77,
).images[0]
image.save(f"canny-8b.jpg")
