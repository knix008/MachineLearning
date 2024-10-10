import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from transformers.utils import logging
logging.set_verbosity_error() 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample, return_timestamps=True)
print("=====================================")
print("The Automatically Recognized Sound!!!")
print("=====================================")
print(result["text"])
print("=====================================")