from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 모델과 토크나이저 로드
model_name = "gpt2"  # 또는 "gpt2-medium", "gpt2-large", "gpt2-xl" 등
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# pad_token_id를 eos_token_id로 설정
tokenizer.pad_token = tokenizer.eos_token

# 텍스트 생성 함수
def generate_response(input_text):
    # 입력 텍스트 토큰화
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # attention_mask와 pad_token_id를 명시적으로 설정
    attention_mask = inputs["attention_mask"]
    pad_token_id = tokenizer.pad_token_id

    # 모델을 사용하여 텍스트 생성
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=attention_mask,  # attention_mask를 명시적으로 전달
        pad_token_id=pad_token_id,  # pad_token_id를 명시적으로 전달
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
    )
    print(outputs)
    # 생성된 텍스트 반환
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "main__":
    response = generate_response("Tell me about korea")
    print(response)