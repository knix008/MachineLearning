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


def main():
    sample = "llava_v1_5_radar.jpg"
    image = Image.open(sample)
    prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

    device = initialize_device()
    model, processor = initialize_model(device)

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    start = time.time()
    output = model.generate(**inputs, max_new_tokens=500)
    end = time.time()

    print(processor.decode(output[0], skip_special_tokens=True))
    print("> The elapsed time : ", end - start)


if __name__ == "__main__":
    main()
