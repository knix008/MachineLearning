import pandas as pd
import csv

train_file_path = "./data/train.csv"
test_file_path = "./data/test.csv"

train_df = pd.read_csv(train_file_path)
train_df = train_df.rename(columns={"sentiment": "label"})
train_df = train_df.reset_index()

test_df = pd.read_csv(test_file_path)
test_df = test_df.rename(columns={"sentiment": "label"})
test_df = test_df.reset_index()

train_data = []
test_data = []

# train_data 초기화
for index, line in train_df.iterrows():
    original_dict = {"text": [], "label": ""}

    if len(line) < 2:
        continue

    original_dict["text"] = line["text"].split(" ")
    original_dict["label"] = line["label"]
    train_data.append(original_dict)

# test_data 초기화
for index, line in test_df.iterrows():
    original_dict = {"text": [], "label": ""}

    if len(line) < 2:
        continue

    original_dict["text"] = line["text"].split(" ")
    original_dict["label"] = line["label"]
    test_data.append(original_dict)

import string

for example in train_data:
    text = [x.lower() for x in example["text"]]
    text = [x.replace("<br", "") for x in text]
    text = ["".join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    example["text"] = text

import random
from sklearn.model_selection import train_test_split

train_data, valid_data = train_test_split(
    train_data, random_state=random.seed(0), test_size=0.2
)
print(f"Number of training examples : {len(train_data)}")
print(f"Number of valid_data examples : {len(valid_data)}")
print(f"Number of test examples : {len(test_data)}")
