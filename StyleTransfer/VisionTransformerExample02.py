import gradio as gr
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import time

model_id = "llava-hf/llava-1.5-7b-hf"
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_caption(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    prompt = "<image> What is shown in this image?\n"  # 반드시 <image> 태그 포함!
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,  # num_beams를 1로 설정하여 greedy decoding 사용
            temperature=1.0,  # temperature를 1.0으로 설정하여 다양성 감소
            top_p=1.0,  # top_p를 1.0으로 설정하여 전체 확률 분포 사용
            top_k=50,  # top_k를 50으로 설정하여 상위 50개 토큰만 고려  
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    elapsed_time = time.time() - start_time
    return f"설명: {caption}\n\n걸린 시간: {elapsed_time:.2f}초"

demo = gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="이미지 설명 및 소요 시간"),
    title="LLaVA 기반 이미지 설명 생성기",
    description="LLaVA를 사용해서 이미지를 설명하고, 걸린 시간(초)을 출력합니다."
)

if __name__ == "__main__":
    demo.launch()