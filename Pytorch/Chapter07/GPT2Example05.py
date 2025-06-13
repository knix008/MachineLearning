from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def beam_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device : ", device)
    torch.manual_seed(799)
    tkz = GPT2Tokenizer.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    cue = "They"
    tkz.pad_token = tkz.eos_token

    input = tkz(cue, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = input["attention_mask"]
    op_beam = mdl.generate(
        input["input_ids"],
        max_length=5,
        pad_token_id=tkz.eos_token_id,
        attention_mask=attention_mask,
        num_beams=3,
        num_return_sequences=3,
        early_stopping=True
    )
    for op_beam_cur in op_beam:
        print(tkz.decode(op_beam_cur, skip_special_tokens=True))


if __name__ == "__main__":
    print("> Beam Search")
    seq = beam_search()
