import os
# Disable Hugging Face Warning Messages.
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'True'

import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from datasets import load_dataset

klue_tc_train = load_dataset('klue', 'ynat', split='train')
klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

print(klue_tc_train[0])
print(klue_tc_train.features['label'].names)
# ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])
#print(klue_tc_train)
#print(klue_tc_train.features['label'])
# ClassLabel(names=['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치'], id=None)
#print(klue_tc_train.features['label'].int2str(1))
# '경제'

klue_tc_label = klue_tc_train.features['label']

def make_str_label(batch):
  batch['label_str'] = klue_tc_label.int2str(batch['label'])
  return batch

klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)
#print(klue_tc_train[0])
# {'title': '유튜브 내달 2일까지 크리에이터 지원 공간 운영', 'label': 3, 'label_str': '생활문화'}

train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
test_dataset = dataset['test']
valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']

def tokenize_function(examples):
    return tokenizer(examples["title"], padding="max_length", truncation=True)

cache_dir = "cache"
model_id = "klue/roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                                                           num_labels=len(train_dataset.features['label'].names), 
                                                           cache_dir=cache_dir, 
                                                           force_download=False)
tokenizer = AutoTokenizer.from_pretrained(model_id)

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    learning_rate=5e-5,
    push_to_hub=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Training and Evaluation
trainer.train()
trainer.evaluate(test_dataset)