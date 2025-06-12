import sys
sys.dont_write_bytecode = True

import torch
from torch.utils.data import Dataset
from PennTreeBank import build_vocab


class PennTreebankDataset(Dataset):
    def __init__(self, data_path, tokenizer, vocab, seq_length):
        self.data = self.load_and_process_data(data_path, tokenizer, vocab, seq_length)

    def load_and_process_data(self, data_path, tokenizer, vocab, seq_length):
        data = []
        with open(data_path, "r") as f:
            for line in f:
                tokens = tokenizer(line.strip())
                token_ids = [
                    vocab.get(token, vocab["<unk>"]) for token in tokens
                ]  # <unk> for unknown
                if len(token_ids) > seq_length:
                    token_ids = token_ids[:seq_length]
                else:
                    token_ids += [vocab["<pad>"]] * (
                        seq_length - len(token_ids)
                    )  # <pad> for padding
                data.append(torch.tensor(token_ids))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


from torch.utils.data import DataLoader

train_data_path = "./ptbdataset/ptb.train.txt"
vocab = build_vocab(train_data_path)
seq_length = 30  # Example sequence length
tokenizer = lambda x: x.split()  # Simple whitespace tokenizer
batch_size = 32  # Example batch size

# Example usage:
train_dataset = PennTreebankDataset(train_data_path, tokenizer, vocab, seq_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader:
for batch in train_loader:
    # Process the batch
    pass
