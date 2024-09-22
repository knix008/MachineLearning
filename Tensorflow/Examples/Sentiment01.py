import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check tensorflow version and GPU availability
print("TensorFlow version : ", tf.__version__)
print("GPU device : ", tf.config.list_physical_devices("GPU"))
print("Numpy version : ", np.__version__)

max_len = 200
n_words = 10000
dim_embedding = 256
EPOCHS = 20
BATCH_SIZE =500

def load_data():
	#load data
	(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=n_words)
	# Pad sequences with max_len
	X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=max_len)
	X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=max_len)
	return (X_train, y_train), (X_test, y_test)

def build_model():
	model = models.Sequential()
	#Input - Emedding Layer
	# the model will take as input an integer matrix of size (batch, input_length)
	# the model will output dimension (input_length, dim_embedding)
    # the largest integer in the input should be no larger
    # than n_words (vocabulary size).
	model.add(layers.Embedding(n_words, 
		dim_embedding, input_length=max_len))

	model.add(layers.Dropout(0.3))

	#takes the maximum value of either feature vector from each of the n_words features
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))

	return model

(X_train, y_train), (X_test, y_test) = load_data()
model=build_model()
model.summary()

model.compile(optimizer = "adam", loss = "binary_crossentropy",
 metrics = ["accuracy"]
)

history = model.fit(X_train, y_train,
 epochs= EPOCHS,
 batch_size = BATCH_SIZE,
 validation_data = (X_test, y_test)
)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
