from transformers import BertTokenizerFast, GPT2LMHeadModel
import torch


def top_p_sampling():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("> Using device : ", device)
    tokenizer = BertTokenizerFast.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2")
    model = GPT2LMHeadModel.from_pretrained("kykim/gpt3-kor-small_based_on_gpt2").to(device)
    cue = "그들은"
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    input = tokenizer(cue, return_tensors="pt", padding=True, truncation=True).to(device)
    attention_mask = input["attention_mask"]

    for i in range(3):
        torch.manual_seed(i + 10)
        output_top_p = model.generate(
            input["input_ids"],
            max_length=10,
            top_p=0.75,
            top_k=0,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            do_sample=True,
        )

        sequence = tokenizer.decode(output_top_p[0], skip_special_tokens=True)
        print(f"> Output {i + 1}: {sequence}")


if __name__ == "__main__":
    print("> Top-p Sampling")
    top_p_sampling()
