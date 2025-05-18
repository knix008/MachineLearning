from huggingface_hub import hf_api

models = hf_api.list_models()
print(models)
num_of_models = len([t for t in models])
print(num_of_models)