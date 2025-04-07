from transformers import pipeline
import torch
import time
import datetime

def run_gemma3():
    pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device="cuda", torch_dtype=torch.bfloat16)

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
            },
        ],
    ]

    start = time.time()
    output = pipe(messages, max_new_tokens=50)
    end = time.time()
    duration = end - start
    print(datetime.timedelta(seconds=duration))
    print(output)

if __name__ == "__main__":
    run_gemma3()