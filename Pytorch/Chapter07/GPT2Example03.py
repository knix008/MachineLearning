from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def greedy_search1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device : ", device)
    torch.manual_seed(799)
    tkz = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    ln = 10
    cue = "They"
    gen = tkz(cue, return_tensors="pt").to(device)
    to_ret = gen["input_ids"][0]

    for i in range(ln):
        outputs = mdl(**gen)
        next_token_logits = torch.argmax(outputs.logits[-1, :])
        to_ret = torch.cat([to_ret, next_token_logits.unsqueeze(0)])
        gen = {"input_ids": to_ret}
    seq = tkz.decode(to_ret)
    return seq


def greedy_search2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device : ", device)
    torch.manual_seed(799)
    tkz = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    ln = 11
    cue = "They"
    tkz.pad_token = tkz.eos_token

    input = tkz(cue, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = input["attention_mask"]
    op_greedy = mdl.generate(
        input["input_ids"],
        max_length=ln,
        pad_token_id=tkz.eos_token_id,
        attention_mask=attention_mask,
    )
    seq = tkz.decode(op_greedy[0], skip_special_tokens=True)
    return seq


if __name__ == "__main__":
    print("> Greedy Search 1")
    seq = greedy_search1()
    print("> The output : ", seq)
    print("> Greedy Search 2")
    seq = greedy_search2()
    print("> The output : ", seq)
