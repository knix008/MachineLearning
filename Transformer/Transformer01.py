import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_text
import tensorflow_datasets as tfds
import tensorflow as tf
import os

logging.getLogger('tensorflow').setLevel(logging.ERROR) # suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Supporting GPU : ", tf.test.is_built_with_cuda())

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True, shuffle_files=True)
train_examples, val_examples = examples['train'], examples['validation']

model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(f'{model_name}.zip', f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)

for pt_examples, en_examples in train_examples.batch(3).take(1):
    print('> Examples in Portuguese:')
for en in en_examples.numpy():
    print(en.decode('utf-8'))

print("> Tokenized...")
encoded = tokenizers.en.tokenize(en_examples)
for row in encoded.to_list():
    print(row)

print("> Original Value...")
round_trip = tokenizers.en.detokenize(encoded)
for line in round_trip.numpy():
    print(line.decode('utf-8'))

MAX_TOKENS=128
def filter_max_tokens(pt, en):
    num_tokens = tf.maximum(tf.shape(pt)[1],tf.shape(en)[1])
    return num_tokens < MAX_TOKENS

def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()
    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
    return (
    ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    .filter(filter_max_tokens)
    .prefetch(tf.data.AUTOTUNE))

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)