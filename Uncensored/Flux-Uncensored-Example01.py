import warnings
import logging
import os
from datetime import datetime
from diffusers import AutoPipelineForText2Image
import torch

warnings.filterwarnings("ignore", message=".*No LoRA keys associated.*")
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Load the base model
pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)

# Load the uncensored LoRA weights
pipeline.load_lora_weights(
    "enhanceaiteam/Flux-uncensored-v2", weight_name="lora.safetensors", prefix=None
)

pipeline.enable_sequential_cpu_offload()

prompt = "Realistic,RAW photo,Sony Alpha A7 II Mirrorless Digital Camera,Canon EF 24-70mm f/2.8L II USM Lens,The Selens 5-in-1 39.4â³ Triangle Reflector,INTRICATE,ATTRACTIVE,FEMALE,WOMAN,Blond Hair,(Any Age),((2girl)),Blond hair,Hardcore-sex,Horney,Home alone,(In bedroom),((in any sex pose)),(open legs showing pussy),(having an orgasm),gorgeous girls,cute girls,girls orgasm,cute,hentai,best quality,(NSFW),(masterpiece),(breasts:0.984),pretty tits,((nipples)),light nipples,cute nipples,normal nipples,no clothes,naked,(messy hair:0.989),(blond fantasy princess hair),(natural hair),(wet wavy hair),(cleavage:0.586),collarbone:0.746),(eyebrows visible through hair:0.732),detailed gorgeous face,(sloppy),saliva,spit,(female orgasm),trembling,embarrassed,(beautiful eyes:0.944),(long hair:0.982),(ponytail:0.741),(sexy),(emo),wide hips,((white lace stockings) ),spreading her pussy,((good pussy)),(juicy pussy),pink pussy,(open pussy),(pussy hair),vagina,camel toe,((clit)),no underwear,indoors,mood lighting,intricate,elegant,perfect hand,perfect fingers <lora:zPDXL2-neg:0.8> zPDXL2 <lora:zPDXL2:0.8> zPDXL2"

# Generate an image with an uncensored NSFW prompt
image = pipeline(prompt=prompt).images[0]
print("Successfully generated an image with the uncensored LoRA weights!")
image.show()

script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_name = (
    torch.cuda.get_device_name(0).replace(" ", "-")
    if torch.cuda.is_available()
    else "cpu"
)
filename = f"{script_name}_{timestamp}_{gpu_name}_bf16_seqoffload.jpg"

image.save(filename)
print(f"Saved: {filename}")
