from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def top_p_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device : ", device)
    tkz = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    cue = "They"
    tkz.pad_token = tkz.eos_token

    input = tkz(cue, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = input["attention_mask"]
    for i in range(3):
        torch.manual_seed(i + 10)
        op_top_k = mdl.generate(
            input["input_ids"],
            max_length=5,
            top_p=0.75,
            top_k=0,
            pad_token_id=tkz.eos_token_id,
            attention_mask=attention_mask,
            do_sample=True,
        )

        seq = tkz.decode(op_top_k[0], skip_special_tokens=True)
        print(f"> Output {i + 1}: {seq}")


if __name__ == "__main__":
    print("> Top-P Sampling")
    top_p_sampling()
