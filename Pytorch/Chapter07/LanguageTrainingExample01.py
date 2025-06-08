import math
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

torch.use_deterministic_algorithms(True)


class Transformer(nn.Module):
    def __init__(
        self, num_token, num_inputs, num_heads, num_hidden, num_layers, dropout=0.3
    ):
        super(Transformer, self).__init__()
        self.model_name = "transformer"
        self.position_enc = PosEnc(num_inputs, dropout)
        layers_enc = TransformerEncoderLayer(num_inputs, num_heads, num_hidden, dropout)
        self.enc_transformer = TransformerEncoder(layers_enc, num_layers)
        self.enc = nn.Embedding(num_token, num_inputs)
        self.num_inputs = num_inputs
        self.dec = nn.Linear(num_inputs, num_token)
        self.init_params()

    def init_params(self):
        initial_rng = 0.12
        self.enc.weight.data.uniform_(-initial_rng, initial_rng)
        self.dec.bias.data.zero_()
        self.dec.weight.data.uniform_(-initial_rng, initial_rng)

    def forward(self, source, mask_source):
        source = self.enc(source) * math.sqrt(self.num_inputs)
        source = self.position_enc(source)
        op = self.enc_transformer(source, mask_source)
        op = self.dec(op)
        return op


def gen_sqr_nxt_mask(size):
    msk = torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1)
    return msk


class PosEnc(nn.Module):
    def __init__(self, d_m, dropout=0.2, size_limit=5000):
        super(PosEnc, self).__init__()
        self.dropout = nn.Dropout(dropout)
        p_enc = torch.zeros(size_limit, 1, d_m)
        pos = torch.arange(size_limit, dtype=torch.float).unsqueeze(1)
        divider = torch.exp(
            torch.arange(0, d_m, 2).float() * (-math.log(10000.0) / d_m)
        )
        p_enc[:, 0, 0::2] = torch.sin(pos * divider)
        p_enc[:, 0, 1::2] = torch.cos(pos * divider)
        self.register_buffer("p_enc", p_enc)

    def forward(self, x):
        return self.dropout(x + self.p_enc[: x.size(0)])


tr_iter = PennTreebank(split="train")
tkzer = get_tokenizer("basic_english")
vocabulary = build_vocab_from_iterator(map(tkzer, tr_iter), specials=["<unk>"])
vocabulary.set_default_index(vocabulary["<unk>"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_data(raw_text):
    numericalised_text = [
        torch.tensor(vocabulary(tkzer(text)), dtype=torch.long) for text in raw_text
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, numericalised_text)))


tr_iter, val_iter, te_iter = PennTreebank()
training_text = process_data(tr_iter)
validation_text = process_data(val_iter)
testing_text = process_data(te_iter)


def gen_batches(text_dataset, batch_size):
    num_batches = text_dataset.size(0) // batch_size
    text_dataset = text_dataset[: num_batches * batch_size]
    text_dataset = text_dataset.view(batch_size, num_batches).t().contiguous()
    return text_dataset.to(device)


training_batch_size = 32
evaluation_batch_size = 16

training_data = gen_batches(training_text, training_batch_size)
validation_data = gen_batches(validation_text, evaluation_batch_size)
testing_data = gen_batches(testing_text, evaluation_batch_size)

max_seq_len = 64


def return_batch(src, k):
    sequence_length = min(max_seq_len, len(src) - 1 - k)
    sequence_data = src[k : k + sequence_length]
    sequence_label = src[k + 1 : k + 1 + sequence_length].reshape(-1)
    return sequence_data, sequence_label


num_tokens = len(vocabulary)  # vocabulary size
embedding_size = 256  # dimension of embedding layer
num_hidden_params = 256  # transformer encoder's hidden (feed forward) layer dimension
num_layers = 2  # num of transformer encoder layers within transformer encoder
num_heads = 2  # num of heads in (multi head) attention models
dropout = 0.25  # value (fraction) of dropout
loss_func = nn.CrossEntropyLoss()
lrate = 4.0  # learning rate
transformer_model = Transformer(
    num_tokens, embedding_size, num_heads, num_hidden_params, num_layers, dropout
).to(device)
optim_module = torch.optim.SGD(transformer_model.parameters(), lr=lrate)
sched_module = torch.optim.lr_scheduler.StepLR(optim_module, 1.0, gamma=0.88)


def train_model():
    transformer_model.train()
    loss_total = 0.0
    time_start = time.time()
    mask_source = gen_sqr_nxt_mask(max_seq_len).to(device)
    num_batches = len(training_data) // max_seq_len
    for b, i in enumerate(range(0, training_data.size(0) - 1, max_seq_len)):
        train_data_batch, train_label_batch = return_batch(training_data, i)
        sequence_length = train_data_batch.size(0)
        if sequence_length != max_seq_len:  # only on last batch
            mask_source = mask_source[:sequence_length, :sequence_length]
        op = transformer_model(train_data_batch, mask_source)
        loss_curr = loss_func(op.view(-1, num_tokens), train_label_batch)
        optim_module.zero_grad()
        loss_curr.backward()
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 0.6)
        optim_module.step()

        loss_total += loss_curr.item()
        interval = 100
        if b % interval == 0 and b > 0:
            loss_interval = loss_total / interval
            time_delta = time.time() - time_start
            print(
                f"epoch {ep}, {b}/{len(training_data)//max_seq_len} batches, training loss {loss_interval:.2f}, training perplexity {math.exp(loss_interval):.2f}"
            )
            loss_total = 0
            time_start = time.time()


def eval_model(eval_model_obj, eval_data_source):
    eval_model_obj.eval()
    loss_total = 0.0
    mask_source = gen_sqr_nxt_mask(max_seq_len).to(device)
    with torch.no_grad():
        for j in range(0, eval_data_source.size(0) - 1, max_seq_len):
            eval_data, eval_label = return_batch(eval_data_source, j)
            sequence_length = eval_data.size(0)
            if sequence_length != max_seq_len:
                mask_source = mask_source[:sequence_length, :sequence_length]
            op = eval_model_obj(eval_data, mask_source)
            op_flat = op.view(-1, num_tokens)
            loss_total += sequence_length * loss_func(op_flat, eval_label).item()
    return loss_total / (len(eval_data_source) - 1)


min_validation_loss = float("inf")
eps = 5
best_model_so_far = None

for ep in range(1, eps + 1):
    ep_time_start = time.time()
    train_model()
    validation_loss = eval_model(transformer_model, validation_data)
    print()
    print(
        f"epoch {ep:}, validation loss {validation_loss:.2f}, validation perplexity {math.exp(validation_loss):.2f}"
    )
    print()

    if validation_loss < min_validation_loss:
        min_validation_loss = validation_loss
        best_model_so_far = transformer_model

    sched_module.step()


testing_loss = eval_model(best_model_so_far, testing_data)
print(
    f"testing loss {testing_loss:.2f}, testing perplexity {math.exp(testing_loss):.2f}"
)
