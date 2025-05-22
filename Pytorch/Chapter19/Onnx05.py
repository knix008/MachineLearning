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

model_onnx = ORTModelForSequenceClassification.from_pretrained("bert-base-uncased", export=True)
model_onnx(**input_ids)

model_onnx.save_pretrained(onnx_directory)
tokenizer.save_pretrained(onnx_directory)

from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer

qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(model_onnx)
quantizer.quantize(save_dir=onnx_directory, quantization_config=qconfig)

model_quantized = ORTModelForSequenceClassification.from_pretrained(onnx_directory, file_name="model.onnx")
model_quantized(**input_ids)
print("> Loading quantized Onnx model...")