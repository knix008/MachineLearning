import tensorflow as tf
import matplotlib.pyplot as plt

class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cn1 = tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1))
        self.cn2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.dp1 = tf.keras.layers.Dropout(0.10)
        self.dp2 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.cn1(x)
        x = self.cn2(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = self.dp1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dp2(x)
        x = self.fc2(x)
        return x
    
# Load the MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channels dimension (required for CNN)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Create a dataloader for training.
train_dataloader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataloader = train_dataloader.shuffle(10000)
train_dataloader = train_dataloader.batch(32)

# Create a dataloader for testing.
test_dataloader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataloader = test_dataloader.batch(500)

tf.random.set_seed(0)
model = ConvNet()
optimizer = tf.keras.optimizers.experimental.Adadelta(learning_rate=0.5)
model.compile(optimizer=optimizer,
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# Train the model using the tf.data.Dataset API
model.fit(train_dataloader, epochs=3, validation_data=test_dataloader)

test_samples = enumerate(test_dataloader)
b_i, (sample_data, sample_targets) = next(test_samples)
plt.imshow(sample_data[0], cmap='gray', interpolation='none')
plt.show()