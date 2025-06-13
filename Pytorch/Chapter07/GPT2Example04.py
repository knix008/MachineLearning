import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("> Using device : ", device)

# Load tokenizer and pretrained model
model_name = "gpt2"  # "gpt2-medium", "gpt2-large", "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set pad_token_id and eos_token_id
tokenizer.pad_token = tokenizer.eos_token


# Text generation
def generate_response(input_text):

    inputs = tokenizer(
        input_text, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    attention_mask = inputs["attention_mask"]
    pad_token_id = tokenizer.pad_token_id

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    print("> Testing GPT2")
    start = time.time()
    response = generate_response("Hi, could you tell me about Korea?")
    end = time.time()
   
    print(response)
    print("> Total time elapsed : ", datetime.timedelta(seconds=end - start))
