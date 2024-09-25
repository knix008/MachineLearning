import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pylab as plt
import tensorflow as tf

print(tf.__version__)

import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image

import tensorflow.compat.v1.keras.backend as K
tf.compat.v1.disable_eager_execution()


classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

# wrap the hub to work with tf.keras

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
grace_hopper = np.array(grace_hopper)/255.0
result = classifier.predict(grace_hopper[np.newaxis, ...])
predicted_class = np.argmax(result[0], axis=-1)

print ("The predicted Class : ", predicted_class)