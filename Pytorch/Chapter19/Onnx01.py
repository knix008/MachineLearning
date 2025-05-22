from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

# a pre-trained model from HuggingFace
model_name = "bert-base-uncased"
onnx_directory = "bert-base-uncased_onnx"

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", export=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

input_ids = tokenizer("I love PyTorch!", return_tensors="pt")
model(**input_ids)

