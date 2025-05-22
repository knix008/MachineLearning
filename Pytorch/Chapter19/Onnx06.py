from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

onnx_directory = "bert-base-uncased_onnx"

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", export=True)

model_quantized = ORTModelForSequenceClassification.from_pretrained(onnx_directory, file_name="model_quantized.onnx")
print("> Loading quantized Onnx model...")
input_ids = tokenizer("I love PyTorch!", return_tensors="pt")
print(model_quantized(**input_ids))