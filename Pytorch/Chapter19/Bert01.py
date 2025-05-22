from transformers import AutoTokenizer, BertForPreTraining
import os
import torch

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits

print(outputs)