import torch
import os
from datetime import datetime
from diffusers import ZImagePipeline

# Load the pipeline
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

# Generate image
prompt = "The image is a high-quality,photorealistic cosplay portrait of a young Asian woman with a soft, idol aesthetic.Physical Appearance: Face: She has a fair,clear complexion.She is wearing striking bright blue contact lenses that contrast with her dark hair.Her expression is innocent and curious,looking directly at the camera with her index finger lightly touching her chin.Hair: She has long,straight jet-black hair with thick,straight-cut bangs (fringe) that frame her face.Attire (Blue & White Bunny Theme): Headwear: She wears tall,upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base,accented with a small white bow.Outfit: She wears a unique blue denim-textured bodysuit.It features a front zipper,silver buttons,and thin silver chains draped across the chest.The sides are constructed from semi-sheer white lace.Accessories: Around her neck is a blue bow tie attached to a white collar.She wears long,white floral lace fingerless sleeves that extend past her elbows,finished with blue cuffs and small black decorative ribbons.Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows.Pose: She is sitting gracefully on the edge of a light-colored,vintage-style bed or cushioned bench.Her body is slightly angled toward the camera,creating a soft and inviting posture.Setting & Background: Location: A bright,high-key studio set designed to look like a clean,airy bedroom.Background: The background is dominated by large windows with white vertical blinds or curtains,allowing soft,diffused natural-looking light to flood the scene.The background is softly blurred (bokeh).Lighting: The lighting is bright,soft,and even,minimizing harsh shadows and giving the skin a glowing,porcelain appearance.Flux Prompt Prompt: A photorealistic,high-quality cosplay portrait of a beautiful Asian woman dressed in a blue and white bunny girl outfit.She has long straight black hair with hime-cut bangs and vibrant blue eyes.She wears tall blue bunny ears with white lace trim,a blue denim-textured bodysuit with a front zipper and white lace side panels,a blue bow tie,and long white lace sleeves.She is sitting on a white bed in a bright,sun-drenched room with soft-focus white curtains.She poses with a finger to her chin,looking at the camera with a soft,innocent expression.8k resolution,high-key lighting,cinematic soft focus,detailed textures of denim and lace,gravure photography style.Key Stylistic Keywords Blue bunny girl,denim cosplay,white lace,high-key lighting,blue contact lenses,black hair with bangs,fishnet stockings,airy atmosphere,photorealistic,innocent and alluring,studio photography."

negative_prompt = "extra hands,extra legs,extra feet,extra arms,Waist Pleats,paintings,sketches,(worst quality:2),(low quality:2),(normal quality:2),lowres,normal quality,((monochrome)),((grayscale)),skin spots,wet,acnes,skin blemishes,age spot,manboobs,backlight,mutated hands,(poorly drawn hands:1.33),blurry,(bad anatomy:1.21),(bad proportions:1.33),extra limbs,(disfigured:1.33),(more than 2 nipples:1.33),(missing arms:1.33),(extra legs:1.33),(fused fingers:1.61),(too many fingers:1.61),(unclear eyes:1.33),lowers,bad hands,missing fingers,extra digit,(futa:1.1),bad hands,missing fingers,(cleft chin:1.3),exposed nipples" # Optional, but would be powerful when you want to remove some unwanted content

# https://prompthero.com/prompt/16c71686861-z-image-turbo-the-image-is-a-high-quality-photorealistic-cosplay-portrait-of-a-young-asian-woman-with-a-soft-idol-aesthetic-physical

height = 1536
width = 1024
cfg_normalization = True
num_inference_steps = 13
guidance_scale = 1.3
seed = 24454331372687

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    cfg_normalization=cfg_normalization,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator("cuda").manual_seed(seed),
).images[0]

script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{script_name}_{timestamp}_{width}x{height}_gs{guidance_scale}_step{num_inference_steps}_cfgnorm{cfg_normalization}_seed{seed}.png"
image.save(filename)
print(f"이미지가 저장되었습니다: {filename}")