from diffusers import DiffusionPipeline
import torch

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

# If you want to compile the refiner's UNet, uncomment the line below
# Note that this may increase memory usage.
# Uncommenting this line is optional and may not be necessary for all use cases.
# If you do not want to compile the UNet, you can skip this step.
#base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

base.to("cuda")
# Enable CPU offloading if needed
#base.enable_model_cpu_offload()

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# If you want to compile the refiner's UNet, uncomment the line below
# Note that this may increase memory usage.
# Uncommenting this line is optional and may not be necessary for all use cases.
# If you do not want to compile the UNet, you can skip this step.
#refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

refiner.to("cuda")
# Enable CPU offloading if needed
#refiner.enable_model_cpu_offload()



# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image.save("lion_refined.png")
