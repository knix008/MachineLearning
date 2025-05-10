import os
from datasets import load_dataset
from transformers import AutoTokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors="tf")
# convert the tokenized datasets to TensorFlow datasets
tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"], label_cols=["labels"], shuffle=True, collate_fn=data_collator, batch_size=8,)