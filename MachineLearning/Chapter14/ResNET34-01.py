import tensorflow as tf
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import pathlib
#import PIL

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

BATCH_SIZE = 32
HEIGHT= 224
WIDTH = 224
EPOCHS = 5

# Check training dataset directory
data_dir = pathlib.Path('flowers/train')
image_count = len(list(data_dir.glob('*/*.jpg')))
print("The numbe of training images : ", image_count)

# Read train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, 
    seed=123,
    image_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE
)

data_dir = pathlib.Path('flowers/validation')
image_count = len(list(data_dir.glob('*/*.jpg')))
print("The numbe of validation images : ", image_count)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(HEIGHT, WIDTH),
  batch_size=BATCH_SIZE
)

data_dir = pathlib.Path('flowers/test')
image_count = len(list(data_dir.glob('*/*.jpg')))
print("The numbe of test images : ", image_count)

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(HEIGHT, WIDTH),
  batch_size=BATCH_SIZE
)

#roses = list(data_dir.glob('roses/*'))
#image = PIL.Image.open(str(roses[1]))
#image.show()

class_names = train_ds.class_names
print(class_names)

# Get some images and show
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

# Get train dataset and it's label for each batch
print("Training DS : ")
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

print("Validation DS : ")
for image_batch, labels_batch in val_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

print("Test DS : ")
for image_batch, labels_batch in test_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

tf.random.set_seed(42)  # extra code – ensures reproducibility

# Define model
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
model = tf.keras.Sequential([
    DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
])

prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation="softmax"))

# Model compile, summary, training, and evaluating
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.summary()

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS
                    )

score = model.evaluate(test_ds)
print("The Scores : ", score[0], score[1])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()