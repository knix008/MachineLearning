from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def main():
    torch.manual_seed(799)
    tkz = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
    ln = 10
    cue = "They"
    gen = tkz(cue, return_tensors="pt")
    to_ret = gen["input_ids"][0]

    for i in range(ln):
        outputs = mdl(**gen)
        next_token_logits = torch.argmax(outputs.logits[-1, :])
        to_ret = torch.cat([to_ret, next_token_logits.unsqueeze(0)])
        gen = {"input_ids": to_ret}
    seq = tkz.decode(to_ret)
    return seq


if __name__ == "__main__":
    seq = main()
    print(seq)
