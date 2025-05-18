from huggingface_hub import hf_api

datasets = hf_api.list_datasets()
print(len([d for d in datasets]))