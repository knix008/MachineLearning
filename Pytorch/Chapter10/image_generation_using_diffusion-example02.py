import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers.utils import make_image_grid
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset
from diffusers import DDPMPipeline
from accelerate import Accelerator
from tqdm import tqdm

DATASET_PATH = "./data/selfie2anime/train/imageB"


def loading_dataset():
    dataset = load_dataset(DATASET_PATH, split="train")
    return dataset


BATCH_SIZE = 16


def load_train_dataset(dataset, batch_size=BATCH_SIZE):
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return train_dataloader


def initialize_noise_scheduler():
    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    return noise_scheduler


def initialize_model():
    # Initialize the UNet model
    model = UNet2DModel(
        sample_size=128,  # the target image resolution
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
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


MODEL_SAVE_DIR = "anime-128"


def initialize_accelerator():
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

    return accelerator


def train(
    model,
    train_dataloader,
    epochs,
    batch_soize,
    noise_scheduler,
    optimizer,
    lr_scheduler,
    accelerator,
):
    global_step = 0
    SAVE_ARTIFACT_EPOCHS = 1
    RANDOM_SEED = 42

    # Now you train the model
    for epoch in range(epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

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

            if (epoch + 1) % SAVE_ARTIFACT_EPOCHS == 0 or epoch == epochs - 1:
                images = pipeline(
                    batch_size=batch_soize,
                    generator=torch.manual_seed(RANDOM_SEED),
                ).images

                # Make a grid out of the images
                image_grid = make_image_grid(images, rows=4, cols=4)

                # Save the images
                test_dir = os.path.join(MODEL_SAVE_DIR, "samples")
                os.makedirs(test_dir, exist_ok=True)
                image_grid.save(f"{test_dir}/{epoch:04d}.png")
                pipeline.save_pretrained(MODEL_SAVE_DIR)


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


def set_lr_scheduler(optimizer, train_dataloader, epochs, num_warmup_steps=500):
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(train_dataloader) * epochs),
    )
    return lr_scheduler


def main():
    NUM_EPOCHS = 3  # Number of epochs to train : 20
    LR = 1e-4
    LR_WARMUP_STEPS = 500

    # Load the model and dataset
    print("> Loading model and dataset...")
    dataset = loading_dataset()
    if dataset is None:
        print("Error: Dataset could not be loaded.")
        return

    print("> Dataset loaded with", len(dataset), "samples")
    print("> Initializing model...")
    model = initialize_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("> Loading train loader...")
    train_dataloader = load_train_dataset(dataset, batch_size=16)

    print("> Setting learning rate scheduler...")
    lr_scheduler = set_lr_scheduler(
        optimizer, train_dataloader, NUM_EPOCHS, LR_WARMUP_STEPS
    )

    print("> Setting dataset transform...")
    dataset.set_transform(transform)

    print("> Initializeing accelerator...")
    accelerator = initialize_accelerator()

    print("> Initializing noise scheduler...")
    noise_scheduler = initialize_noise_scheduler()

    print(
        "> Preparing model, optimizer, dataloader, and lr_scheduler with accelerator..."
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Train the model
    print("> Starting training...")
    train(
        model,
        train_dataloader,
        NUM_EPOCHS,
        16,
        noise_scheduler,
        optimizer,
        lr_scheduler,
        accelerator,
    )


if __name__ == "__main__":
    print("> Image generation using diffusion example")
    main()
