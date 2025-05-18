from huggingface_hub import hf_api

models = hf_api.list_models()
print(models)