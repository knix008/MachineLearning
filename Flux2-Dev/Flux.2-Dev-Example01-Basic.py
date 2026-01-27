import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from datetime import datetime
import os

repo_id = "black-forest-labs/FLUX.2-dev"  # Standard model
torch_dtype = torch.float16  # Use float16 for GPU efficiency

# Load model on GPU with CPU offloading for memory management
pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)

# Enable sequential CPU offload - moves layers to GPU only when needed
# This allows running large models by using GPU for compute and CPU for storage
pipe.enable_sequential_cpu_offload()
print("모델 로딩 완료!")
print(
    "Sequential CPU offloading enabled - model layers will move between GPU and CPU as needed"
)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

# Use the pipe directly - it handles text encoding internally
# Generator device should match where the model's first layer is
device_for_generator = "cuda" if torch.cuda.is_available() else "cpu"
image = pipe(
    prompt=prompt,
    generator=torch.Generator(device=device_for_generator).manual_seed(42),
    num_inference_steps=28,  # 28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

# Save with program name and timestamp
script_name = os.path.splitext(os.path.basename(__file__))[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image.save(f"{script_name}_{timestamp}.png")