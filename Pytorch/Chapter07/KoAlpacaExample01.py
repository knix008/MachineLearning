import torch
from transformers import pipeline, AutoModelForCausalLM

MODEL = "beomi/KoAlpaca-Polyglot-5.8B"


def initialize_ko_alpaca():
    print(f"> Using device: {torch.cuda.get_device_name(0)}")
    print(f"> Model: {MODEL}")
    print("> Initializing KoAlpaca...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        print("> CUDA is available. Using GPU.")
        device = "cuda"
    else:
        print("> CUDA is not available. Using CPU.")
        device = "cpu"
    print(f"> Device set to: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)
    return pipe


def ask(x, context="", is_input_full=False):
    ans = pipe(
        (
            f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:"
            if context
            else f"### 질문: {x}\n\n### 답변:"
        ),
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    return ans[0]["generated_text"]


def run_ko_alpaca_example(pipe):
    print("> KoAlpaca Example")
    print("You can ask questions in Korean. Type 'exit' to quit.")

    while True:
        user_input = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == "exit":
            break
        context = input("맥락을 입력하세요 (없으면 Enter): ")
        is_input_full = False  # Placeholder for future use
        result = ask(user_input, context, is_input_full)
        print(f"답변: {result}")


if __name__ == "__main__":
    print("> KoAlpaca Example")
    pipe = initialize_ko_alpaca()
    run_ko_alpaca_example(pipe)
