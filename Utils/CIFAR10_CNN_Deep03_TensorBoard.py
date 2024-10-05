import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import datetime

EPOCHS=50
NUM_CLASSES = 10
BATCH_SIZE = 128
INPUT_SHAPE = (32, 32, 3)

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
 
    #normalize 
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
 
    y_train =  tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
    y_test =  tf.keras.utils.to_categorical(y_test,NUM_CLASSES)

    return x_train, y_train, x_test, y_test

def build_model(): 
    model = models.Sequential()
    
    #1st blocl
    model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=INPUT_SHAPE, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))

    #2nd block
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    #3d block 
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.4))

    #dense  
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    return model

def plot_result(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    plt.savefig("Training_accuracy.png")
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.show()
    plt.savefig("Training_loss.png")

def main(): 
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Check tensorflow version and GPU availability
    print("TensorFlow version : ", tf.__version__)
    print("GPU device : ", tf.config.list_physical_devices("GPU"))
    print("Numpy version : ", np.__version__)

    (x_train, y_train, x_test, y_test) = load_data()
    
    print("Training input shape : ", x_train.shape)
    print("Training target shape : ", y_train.shape)
    print("Test input shape : ", x_test.shape)
    print("Test target shape : ", y_test.shape)
    
    model = build_model() # 32 x 32 x 3
    model.compile(loss='categorical_crossentropy', 
                optimizer='RMSprop', 
                metrics=['accuracy'])
    model.summary()
    
    #image augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    #train
    batch_size = 64
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(x_test,y_test),
                        callbacks=[tensorboard_callback])
    
    #save to disk
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    #model.save_weights('model.h5') 
    
    #test
    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('\nTest accuracy : %.3f, loss: %.3f' % (scores[1]*100,scores[0])) 
    
    plot_result(history)

if __name__ == "__main__":
    main()