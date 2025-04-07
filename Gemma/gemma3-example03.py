from transformers import pipeline
import torch
import time
import datetime

def run_gemma3():
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-4b-it",
        device="cuda",
        torch_dtype=torch.bfloat16,
        token=""
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }
    ]


    start = time.time()
    output = pipe(text=messages, max_new_tokens=200)
    end = time.time()
    duration = end - start
    print(datetime.timedelta(seconds=duration))

    print(output[0]["generated_text"][-1]["content"])
    # Okay, let's take a look! 
    # Based on the image, the animal on the candy is a **turtle**. 
    # You can see the shell shape and the head and legs.

if __name__ == "__main__":
    run_gemma3()