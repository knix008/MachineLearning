from PIL import Image
import numpy as np

from datasets import load_dataset

dataset = load_dataset("./data/selfie2anime/train/imageB", split="train")
print(dataset)

from diffusers.utils import make_image_grid

make_image_grid(dataset["image"][:16], rows=4, cols=4)


img = dataset["image"][0]
# Show the image
if isinstance(img, Image.Image):
    img.show()


noise = 256 * np.random.rand(*img.size, 3)
noisy_img = ((img + noise) / 2).astype(np.uint8)
Image.fromarray(noisy_img)
Image.fromarray(noise.astype(np.uint8))
more_noisy_img = ((img + 4 * noise) / 5).astype(np.uint8)
Image.fromarray(more_noisy_img)

less_noisy_img = ((img + 0.5 * noise) / 1.5).astype(np.uint8)
Image.fromarray(less_noisy_img)

from torchvision import transforms

IMAGE_SIZE = 128
preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

import torch

BSIZE = 16
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BSIZE, shuffle=True)

from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


clean_images = next(iter(train_dataloader))["images"]
# Sample noise to add to the images
noise = torch.randn(clean_images.shape, device=clean_images.device)
bs = clean_images.shape[0]

# Sample a random timestep for each image
timesteps = torch.arange(10, 161, 10, dtype=torch.int)

# Add noise to the clean images according to the noise magnitude at each timestep
# (this is the forward diffusion process)
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

make_image_grid(
    [transforms.ToPILImage()(clean_image) for clean_image in clean_images],
    rows=4,
    cols=4,
)

make_image_grid(
    [transforms.ToPILImage()(noisy_image) for noisy_image in noisy_images],
    rows=4,
    cols=4,
)


clean_images = next(iter(train_dataloader))["images"]
# Sample noise to add to the images
noise = torch.randn(clean_images.shape, device=clean_images.device)
bs = clean_images.shape[0]

# Sample a random timestep for each image
timesteps = torch.randint(
    0,
    noise_scheduler.config.num_train_timesteps,
    (bs,),
    device=clean_images.device,
    dtype=torch.int64,
)

# Add noise to the clean images according to the noise magnitude at each timestep
# (this is the forward diffusion process)
noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=IMAGE_SIZE,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(
        128,
        128,
        256,
        256,
        512,
        512,
    ),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


from diffusers.optimization import get_cosine_schedule_with_warmup

NUM_EPOCHS = 20
LR = 1e-4
LR_WARMUP_STEPS = 500

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=LR_WARMUP_STEPS,
    num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
)


import os
from accelerate import Accelerator

MODEL_SAVE_DIR = "anime-128"

# Initialize accelerator and tensorboard logging
accelerator = Accelerator(
    mixed_precision="fp16",
    log_with="tensorboard",
    project_dir=os.path.join(MODEL_SAVE_DIR, "logs"),
)

if accelerator.is_main_process:
    if MODEL_SAVE_DIR is not None:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    accelerator.init_trackers("train_example")


model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

from tqdm import tqdm
import torch.nn.functional as F
from diffusers import DDPMPipeline

global_step = 0
SAVE_ARTIFACT_EPOCHS = 1
RANDOM_SEED = 42

# Now you train the model
for epoch in range(NUM_EPOCHS):
    progress_bar = tqdm(
        total=len(train_dataloader), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"]
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
            dtype=torch.int64,
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        global_step += 1

    # After each epoch you optionally sample some demo images with evaluate() and save the model
    if accelerator.is_main_process:
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
        )

        if (epoch + 1) % SAVE_ARTIFACT_EPOCHS == 0 or epoch == NUM_EPOCHS - 1:
            images = pipeline(
                batch_size=BSIZE,
                generator=torch.manual_seed(RANDOM_SEED),
            ).images

            # Make a grid out of the images
            image_grid = make_image_grid(images, rows=4, cols=4)

            # Save the images
            test_dir = os.path.join(MODEL_SAVE_DIR, "samples")
            os.makedirs(test_dir, exist_ok=True)
            image_grid.save(f"{test_dir}/{epoch:04d}.png")

            pipeline.save_pretrained(MODEL_SAVE_DIR)
