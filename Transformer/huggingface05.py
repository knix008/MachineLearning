import os
from transformers import pipeline
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ner_pipe = pipeline("ner")
sequence = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much."""
for entity in ner_pipe(sequence):
    print(entity)