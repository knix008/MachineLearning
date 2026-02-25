from diffusers import AutoPipelineForText2Image
import torch
import datetime

# Load the base model
pipeline = AutoPipelineForText2Image.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)

# Load the uncensored LoRA weights
pipeline.load_lora_weights(
    "enhanceaiteam/Flux-uncensored-v2", weight_name="lora.safetensors", prefix=None
)
pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()
pipeline.to("cpu")

# Generate an image with an uncensored NSFW prompt
image = pipeline("a naked cute girl walking on a sunny beach.").images[0]
image.save(
    f"Flux-Uncensored-Example01-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
)
