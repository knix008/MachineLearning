import gradio as gr
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import time

# 모델 및 토크나이저 로딩
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_caption(image):
    start_time = time.time()
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, max_length=16)  # num_beams 제거
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    elapsed_time = time.time() - start_time
    return f"설명: {caption}\n\n걸린 시간: {elapsed_time:.2f}초"


demo = gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="이미지 설명 및 소요 시간"),
    title="ViT 기반 이미지 설명 생성기",
    description="Vision Transformer(ViT)와 GPT2를 사용해서 이미지를 설명하고, 걸린 시간(초)을 출력합니다.",
)

if __name__ == "__main__":
    demo.launch()
