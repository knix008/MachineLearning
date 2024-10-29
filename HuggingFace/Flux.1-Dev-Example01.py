import os
import warnings
import torch
from diffusers import FluxPipeline

# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
# Disable all other warning message.
warnings.filterwarnings("ignore")
access_token="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"

def run_flux1_dev():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using... : ", device)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token=access_token)
    #pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    pipe.to(device)

    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image.save("flux-dev.png")

def main():
    run_flux1_dev()

if __name__ == "__main__": 
    main()