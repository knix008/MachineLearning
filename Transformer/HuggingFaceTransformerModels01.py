from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Disable warning messages.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

repo = "davidkim205/iris-7b"
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(repo)

def generate(prompt):
    encoding = tokenizer(
        prompt,
        return_tensors='pt',
        return_token_type_ids=False
    ).to("cuda")
    gen_tokens = model.generate(
        **encoding,
        max_new_tokens=2048,
        temperature=1.0,
        num_beams=5,
    )
    prompt_end_size = encoding.input_ids.shape[1]
    result = tokenizer.decode(gen_tokens[0, prompt_end_size:])

    return result


def translate_ko2en(text):
    prompt = f"[INST] 다음 문장을 영어로 번역하세요.{text} [/INST]"
    return generate(prompt)


def translate_en2ko(text):
    prompt = f"[INST] 다음 문장을 한글로 번역하세요.{text} [/INST]"
    return generate(prompt)


def main():
    while True:
        text = input('>')
        en_text = translate_ko2en(text)
        ko_text = translate_en2ko(en_text)
        print('en_text', en_text)
        print('ko_text', ko_text)

if __name__ == "__main__":
    main()