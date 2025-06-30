import gradio as gr
from aura_sr import AuraSR
from PIL import Image
import os
import time

# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

def upscale_image(input_image, upsample_times):
    """
    Upscales the input_image by 4x, upsample_times number of times (e.g., 1 = 4x, 2 = 16x).
    Returns the final image and the elapsed time(s).
    """
    image = input_image
    times = []
    for i in range(upsample_times):
        start = time.time()
        image = aura_sr.upscale_4x_overlapped(image)
        end = time.time()
        elapsed = end - start
        times.append(f"Step {i+1}: {elapsed:.5f} sec")
    time_report = "\n".join(times)
    return image, time_report

title = "AuraSR Image Upscaler"
description = "Upload an image and select how many times you want to upscale it by 4x (e.g., 1 = 4x, 2 = 16x). Model: fal/AuraSR-v2"

demo = gr.Interface(
    fn=upscale_image,
    inputs=[
        gr.Image(type="pil", label="Input Image"), 
        gr.Slider(1, 2, value=1, step=1, label="Upscale Steps (1x4x, 2x16x)")
    ],
    outputs=[
        gr.Image(type="pil", label="Upscaled Image"),
        gr.Textbox(label="Elapsed Time(s) for Each Step")
    ],
    title=title,
    description=description,
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()