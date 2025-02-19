from huggingface_hub import list_datasets

all_datasets = [ds.id for ds in list_datasets()]
print(f"Current dataset : {len(all_datasets)}")
print(f"Fist 10 dataset : {all_datasets[:10]}")