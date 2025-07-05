import torch
from diffusers import FluxPipeline

access_token="hf_"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token=access_token)
pipe.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-live-3D", weight_name="FLUX-dev-lora-live_3D.safetensors")
pipe.fuse_lora(lora_scale=1.1)
pipe.to("cuda")

prompt = "A colorful cartoon monkey sits on a bus as it rolls down the street in Times Square, New York"

image = pipe(prompt, 
             num_inference_steps=24, 
             guidance_scale=3.5,
            ).images[0]
image.save(f"example.png")