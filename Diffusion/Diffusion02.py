from diffusers import StableDiffusionPipeline
import torch

model_id = "kyujinpy/KO-anything-v4.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "1소년, 강아지 귀, 귀여운, 흰색 스카프, 눈, 관찰자"
image = pipe(prompt).images[0]

image.save("./hatsune_miku.png")