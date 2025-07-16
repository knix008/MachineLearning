from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel
import torch

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    text_encoder_3=t5_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()

prompt = "a beautiful skinny woman wearing a high legged red bikini, walking on the sunny beach, photorealistic, 8k resolution, ultra detailed, vibrant colors, highly detailed, cinematic lighting, realistic shadows, depth of field, no bad anatomy, no text, no watermark, no logo, no signature, no low quality, no blurry, no bad quality, no low resolution, no cropped image"

image = pipeline(
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=0.0,
    max_sequence_length=512,
).images[0]
image.save("red-bikini-woman-3.5-large-turbo.png")