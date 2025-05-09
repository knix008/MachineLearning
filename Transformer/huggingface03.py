import os
from transformers import TFAutoModelForSequenceClassification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")