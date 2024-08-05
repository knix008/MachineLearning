import tensorflow as tf
import numpy as np

x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[0],[1],[1],[0]])
print(x.shape)
print(y.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse')
model.summary()

history = model.fit(x, y, epochs=2000, batch_size=1)
model.predict(x)