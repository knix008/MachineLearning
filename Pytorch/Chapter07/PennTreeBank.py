import collections
import numpy as np

train_path = "./data/ptbdataset/ptb.train.txt"
valid_path = "./data/ptbdataset/ptb.valid.txt"
test_path = "./data/ptbdataset/ptb.test.txt"


def read_words(filename):
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = zip(*count_pairs)
    word_to_id = {word: i for i, word in enumerate(words)}
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_ptb_dataset():
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocab_size = len(word_to_id)

    return train_data, valid_data, test_data, vocab_size


def id_to_word(id_list):
    word_to_id = build_vocab(train_path)
    id_to_word_dict = {v: k for k, v in word_to_id.items()}
    return [id_to_word_dict[id_] for id_ in id_list]


def ptb_iterator(raw_data, batch_size, num_steps):
    data_length = len(raw_data)
    batch_length = data_length // batch_size
    data = np.reshape(
        raw_data[0 : batch_size * batch_length], [batch_size, batch_length]
    )

    epoch_size = (batch_length - 1) // num_steps

    for i in range(epoch_size):
        x = data[:, i * num_steps : (i + 1) * num_steps]
        y = data[:, i * num_steps + 1 : (i + 1) * num_steps + 1]
        yield (x, y)

def main():
    train_data, valid_data, test_data, vocab_size = load_ptb_dataset()
    print(
        f"Vocabulary size: {vocab_size}",
        f", Train data size: {len(train_data)}",
        f", Valid data size: {len(valid_data)}",
        f", Test data size: {len(test_data)}",
    )
    print(id_to_word(train_data[0:100]))
    print(train_data[0:100])
    
    batch_size = 30
    num_steps = 20
    iteration = ptb_iterator(train_data, batch_size, num_steps)
    tupl = iteration.__next__()
    input_data = tupl[0]
    targets = tupl[1]
    print(f"Input data shape: {input_data.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Input data: {input_data[0]}")
    print(id_to_word(input_data[0, :]))
    print(f"Targets: {targets[0]}")
    
if __name__ == "__main__":
    main()
