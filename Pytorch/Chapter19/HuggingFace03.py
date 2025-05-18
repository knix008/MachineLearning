from huggingface_hub import hf_api

models = hf_api.list_models()
#print(models)
#num_of_models = len([t for t in models])
#print(num_of_models)

text_gen_models = [model.id for model in models if "text-generation-inference" in model.tags and model.downloads>1000000]
print(text_gen_models)