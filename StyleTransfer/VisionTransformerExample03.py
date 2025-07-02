import gradio as gr
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import time

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=True)
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# pad_token_id를 명시적으로 설정하여 경고 메시지 제거
pad_token_id = processor.tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = processor.tokenizer.eos_token_id
model.config.pad_token_id = pad_token_id


def predict_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    prompt = "[INST] <image> What is shown in this image? [/INST]\n"
    start = time.time()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=pad_token_id,  # pad_token_id를 명시적으로 지정
        )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    end = time.time()
    elapsed_time = end - start
    return f"설명: {caption}\n\n걸린 시간: {elapsed_time:.2f}초"


demo = gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="이미지 설명 및 소요 시간"),
    title="LLaVA 기반 이미지 설명 생성기",
    description="LLaVA를 사용해서 이미지를 설명하고, 걸린 시간(초)을 출력합니다.",
)

if __name__ == "__main__":
    demo.launch()
