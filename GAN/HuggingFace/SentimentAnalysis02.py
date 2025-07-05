# from huggingface_hub import list_datasets

# all_datasets = [ds.id for ds in list_datasets()]
# print(f"Current dataset : {len(all_datasets)}")
# print(f"Fist 10 dataset : {all_datasets[:10]}")

from datasets import load_dataset
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

emotions = load_dataset("emotion")