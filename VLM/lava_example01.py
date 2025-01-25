from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
import torch
import time
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(device)

#url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
file = "Certificate01.png"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(file)
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
inputs = processor(prompt, image, return_tensors="pt").to(device)

start = time.time()
output = model.generate(**inputs, max_new_tokens=1000)
print(processor.decode(output[0], skip_special_tokens=True))
end = time.time()

sec = (end - start)
result = datetime.timedelta(seconds=sec)
result_list = str(datetime.timedelta(seconds=sec)).split(".")
print(result_list[0])