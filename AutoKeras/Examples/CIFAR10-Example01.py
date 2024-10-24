import matplotlib.pyplot as plt
import autokeras as ak
from tensorflow.keras.datasets import cifar10 
import numpy as np
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.utils import plot_model

print("AutoKeras version : ", ak.__version__)

# Prepare the dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape) 
print(x_test.shape) 

train_histogram = np.histogram(y_train)
test_histogram = np.histogram(y_test)
_, axs = plt.subplots(1, 2)
axs[0].set_xticks(range(10))
axs[0].bar(range(10), train_histogram[0])
axs[0].set_title('Train dataset histogram')
axs[1].set_xticks(range(10))
axs[1].bar(range(10), test_histogram[0])
axs[1].set_title('Test dataset histogram')
plt.show()

# Initialize the ImageClassifier.
clf = ak.ImageClassifier(max_trials=1)
# Search for the best model.
clf.fit(x_train, y_train)

metrics = clf.evaluate(x_test, y_test) 
print(metrics) 

input_node = ak.ImageInput()
output_node = ak.EfficientNetBlock(
    # Only use EfficientNetb7 architecture.
    version="b7",
    # Load pretrained ImageNet weights 
    pretrained=True)(input_node)

output_node = ak.ClassificationHead()(output_node)

# Search for the best model with EarlyStopping.
cbs = [tf_callbacks.EarlyStopping(patience=2, verbose=1),]
clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=False,
    max_trials=10,
    objective='val_accuracy')
clf.fit(x_train, y_train, callbacks = cbs, verbose=1, epochs=10)

# Evaluate the chosen model with testing data
print(x_test.shape)
print(y_test.shape)
clf.evaluate(x_test, y_test)

predicted_y = clf.predict(x_test[:10]).argmax(axis=1)
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]
fig = plt.figure(figsize=[18,6])
for i in range(len(predicted_y)):
    ax = fig.add_subplot(2, 5, i+1)
    ax.set_axis_off()
    ax.set_title('Prediced: %s, Real: %s' % (labelNames[int(predicted_y[i])], labelNames[int(y_test[i])]))
    img = x_test[i]
    ax.imshow(img)
plt.show()

# First we export the model to a keras model
model = clf.export_model()

# Now, we ask for the model Sumary:
clf.summary()

plot_model(clf.export_model())