import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from datasets import load_dataset
import transformers as transformers
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import TFAutoModelForSequenceClassification

# NOTICE!!! 
# To run the following code, you need to downgrade your transformer version.
# For example, run "$ pip install transformers==4.37.2" command in your terminal.

print("TensorFlow version : ", tf.__version__)
print("Transformer version : ", transformers.__version__)

dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

data_collator = DefaultDataCollator(return_tensors="tf")

# convert the tokenized datasets to TensorFlow datasets
tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #losss=tf.compat.v1.losses.sparse_softmax_cross_entropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)