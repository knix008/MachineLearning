import torch
import os
from diffusers import AutoPipelineForText2Image


def init_pipeline():
    """
    Initialize the Stable Diffusion pipeline.
    This function sets up the pipeline for text-to-image generation.
    """
    # Disable warning messages.
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Checking GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Device : ", device)

    if device.type == "cuda":
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
        )
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
        )

    return pipeline.to(device)


def generate_image(pipeline, prompt, generator):
    """
    Generate an image from a text prompt using the provided pipeline.

    Args:
        pipeline: The initialized Stable Diffusion pipeline.
        prompt: The text prompt for image generation.
        generator: A list containing a torch.Generator for reproducibility.

    Returns:
        The generated image.
    """
    return pipeline(prompt, generator=generator).images[0]


def setup_generator(device):
    """
    Set up a generator for reproducibility.

    Args:
        device: The device on which the generator will run.

    Returns:
        A list containing a torch.Generator.
    """
    return [torch.Generator(device=device)]


if __name__ == "__main__":
    # Save the generated image
    pipeline = init_pipeline()
    generator = setup_generator(pipeline.device)
    image = generate_image(
        pipeline, "The Taj Mahal on the Moon with Earth in sight.", generator
    )
    # Display the image
    image.show()
    image.save("TajMahalOnMoon.png")
