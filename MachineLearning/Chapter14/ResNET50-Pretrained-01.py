import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

model = tf.keras.applications.ResNet50(weights="imagenet")

images = load_sample_images()["images"]
print("Loaded image shape : ", np.array(images).shape)
images_resized = tf.keras.layers.Resizing(height=224, width=224,
                                          crop_to_aspect_ratio=True)(images)

inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)
print("Input shape : ", inputs.shape)

Y_proba = model.predict(inputs)
print("Prediction shape : ", Y_proba.shape)

top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print(f"Image #{image_index}")
    for class_id, name, y_proba in top_K[image_index]:
        print(f"  {class_id} - {name:12s} {y_proba:.2%}")
        
plt.figure(figsize=(10, 6))
for idx in (0, 1):
    plt.subplot(1, 2, idx + 1)
    plt.imshow(images_resized[idx] / 255)
    plt.axis("off")

plt.show()