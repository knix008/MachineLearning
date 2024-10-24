import os
import torch
from diffusers import StableDiffusionPipeline

# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
import warnings
warnings.filterwarnings("ignore")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using... : ", device)
    
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.to("cuda") # mps for mac
    prompt = "a photo of an astronaut riding a horse on mars, blazing fast, wind and sand moving back"
    image = pipe(
        prompt, num_inference_steps=30
    ).images[0]
    image.save(f"{prompt}.jpg")

if __name__ == "__main__":
    main()