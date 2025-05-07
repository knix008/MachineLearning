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