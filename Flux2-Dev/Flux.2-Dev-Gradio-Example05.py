import torch
from transformers import Mistral3ForConditionalGeneration
from diffusers import Flux2Pipeline, Flux2Transformer2DModel


prompt = "A photorealistic gravure-style full-body portrait of a beautiful young korean woman standing by a large window in a bright white room. She has long dark brown hair and a soft,alluring expression.She is wearing a stylish black lingerie set with mesh details and strappy accents, paired with black fishnet thigh-high stockings. She is standing by a white window seat covered with a white faux fur rug, with one leg tucked under her and the other leg extended down white steps. She leans her elbow on the window sill, touching her hair. The background features sheer white curtains and a blurred city view through the window grid. Bright natural daylight, high-key lighting, realistic skin texture,8k resolution, elegant boudoir aesthetic. Key Stylistic Keywords: High-key lighting, white room,black lingerie, fishnets, window seat, faux fur texture, natural daylight, photorealistic, gravure style, elegant, airy."


# We need to update the package dependencies to the latest versions to run this example:
# pip install git+https://github.com/huggingface/diffusers.git
# pip install --upgrade transformers accelerate bitsandbytes


repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
torch_dtype = torch.bfloat16

transformer = Flux2Transformer2DModel.from_pretrained(
    repo_id,
    subfolder="transformer",
    torch_dtype=torch_dtype,
    device_map="cpu",
)

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder="text_encoder",
    dtype=torch_dtype,
    device_map="cpu",
)

pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    transformer=transformer,
    text_encoder=text_encoder,
    torch_dtype=torch_dtype,
)
pipe.enable_model_cpu_offload()

print("모델 로딩 완료!")

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    generator=torch.manual_seed(42),
).images[0]

image.save("flux2_dev_bnb_4bit_example05.png")
print("이미지 생성 완료!")