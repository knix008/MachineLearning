import torch
from diffusers import Flux2Pipeline
from datetime import datetime
import os

# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]

device = "mps"
device_type = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", torch_dtype=device_type
)

pipe.load_lora_weights(
    "fal/FLUX.2-dev-Turbo", weight_name="flux.2-turbo-lora.safetensors"
)
print("LoRA weights loaded.")

prompt = "The image is a high-quality,photorealistic cosplay portrait of a young Asian woman with a soft, idol aesthetic.Physical Appearance: Face: She has a fair,clear complexion.She is wearing striking bright blue contact lenses that contrast with her dark hair.Her expression is innocent and curious,looking directly at the camera with her index finger lightly touching her chin.Hair: She has long,straight jet-black hair with thick,straight-cut bangs (fringe) that frame her face.Attire (Blue & White Bunny Theme): Headwear: She wears tall,upright blue fabric bunny ears with white lace inner lining and a delicate white lace headband base,accented with a small white bow.Outfit: She wears a unique blue denim-textured bodysuit.It features a front zipper,silver buttons,and thin silver chains draped across the chest.The sides are constructed from semi-sheer white lace.Accessories: Around her neck is a blue bow tie attached to a white collar.She wears long,white floral lace fingerless sleeves that extend past her elbows,finished with blue cuffs and small black decorative ribbons.Legwear: She wears white fishnet stockings held up by blue and white ruffled lace garters adorned with small white bows.Pose: She is sitting gracefully on the edge of a light-colored,vintage-style bed or cushioned bench.Her body is slightly angled toward the camera,creating a soft and inviting posture.Setting & Background: Location: A bright,high-key studio set designed to look like a clean,airy bedroom.Background: The background is dominated by large windows with white vertical blinds or curtains,allowing soft,diffused natural-looking light to flood the scene.The background is softly blurred (bokeh).Lighting: The lighting is bright,soft,and even,minimizing harsh shadows and giving the skin a glowing,porcelain appearance.Flux Prompt Prompt: A photorealistic,high-quality cosplay portrait of a beautiful Asian woman dressed in a blue and white bunny girl outfit.She has long straight black hair with hime-cut bangs and vibrant blue eyes.She wears tall blue bunny ears with white lace trim,a blue denim-textured bodysuit with a front zipper and white lace side panels,a blue bow tie,and long white lace sleeves.She is sitting on a white bed in a bright,sun-drenched room with soft-focus white curtains.She poses with a finger to her chin,looking at the camera with a soft,innocent expression.8k resolution,high-key lighting,cinematic soft focus,detailed textures of denim and lace,gravure photography style.Key Stylistic Keywords Blue bunny girl,denim cosplay,white lace,high-key lighting,blue contact lenses,black hair with bangs,fishnet stockings,airy atmosphere,photorealistic,innocent and alluring,studio photography."


image = pipe(
    prompt=prompt,
    sigmas=TURBO_SIGMAS,
    guidance_scale=2.5,
    height=1024,
    width=1024,
    num_inference_steps=8,
).images[0]

# Generate filename with script name and current date and time
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{script_name}_{timestamp}.png"
image.save(output_filename)
print(f"이미지가 저장되었습니다: {output_filename}")
