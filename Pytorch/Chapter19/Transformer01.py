from transformers import AutoModelForSequenceClassification, AutoTokenizer

# a pre-trained model from HuggingFace
model_name = "bert-base-uncased"
# Load the pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)