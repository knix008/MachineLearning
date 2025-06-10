# This example is from "https://huggingface.co/blog/vlms"
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from PIL import Image
import time
import datetime


def initialize_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def initialize_model(device):
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf", use_fast=True
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return model, processor


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def run_inference(model, processor, image, prompt, device):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(output[0], skip_special_tokens=True)


def main():
    image = load_image("llava_v1_5_radar.jpg")
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    device = initialize_device()
    model, processor = initialize_model(device)

    start = time.time()
    output = run_inference(model, processor, image, prompt, device)
    end = time.time()
    elapsed_time = end - start

    print("> Output : ", output)
    print("> The elapsed time : ", str(datetime.timedelta(seconds=elapsed_time)).split("."))


if __name__ == "__main__":
    main()
