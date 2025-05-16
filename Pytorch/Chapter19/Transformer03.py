from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# a pre-trained model from HuggingFace
model_name = "bert-base-uncased"
# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "I love PyTorch!"

# Tokenize the input text using the tokenizer
inputs = tokenizer(input_text, return_tensors="pt")
# Perform inference using the pre-trained model
with torch.no_grad():
    outputs = model(**inputs)
#print(outputs)

# Access the model predictions or outputs
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print("> The predicted class : ", predicted_class)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)