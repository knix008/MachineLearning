import os
import warnings
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Disable warning messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'
warnings.filterwarnings("ignore")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using... : ", device)
    
    model_id = "stabilityai/stable-diffusion-2-1"
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    prompt = "an astronaut riding a green horse in picasso style"
    image = pipe(prompt).images[0]
    image.save(f"{prompt}.jpg")
    
if __name__ == "__main__":
    main()