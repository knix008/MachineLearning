import numpy as np
from skimage.transform import resize
from imageio.v2 import imread

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD

model_architecture = "model.json"      # This must be saved before.
model_weights = "model.weights.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

img_names = ["cat-standing.jpg", "dog.jpg"]
imgs = [resize(imread(img_name), (32, 32)).astype("float32") for img_name in img_names]
imgs = np.array(imgs) / 255
print("imgs.shape:", imgs.shape)

optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

predictions = model.predict(imgs)
print("predictions:", predictions)