from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def top_k_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device : ", device)
    torch.manual_seed(799)
    tkz = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    cue = "They"
    tkz.pad_token = tkz.eos_token

    input = tkz(cue, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = input["attention_mask"]
    for i in range(3):
        op_top_k = mdl.generate(
            input["input_ids"],
            max_length=5,
            pad_token_id=tkz.eos_token_id,
            attention_mask=attention_mask,
            do_sample=True,
            top_k=2
        )
        
        seq = tkz.decode(op_top_k[0], skip_special_tokens=True)
        print(f"> Output {i + 1}: {seq}")

if __name__ == "__main__":
    print("> Top K Sampling")
    top_k_sampling()
