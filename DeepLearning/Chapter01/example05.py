import tensorflow as tf
from tensorflow import keras

labels = [ 0, 2, 1, 2, 0]
print(tf.keras.utils.to_categorical(labels))