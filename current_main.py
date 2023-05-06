import keras.callbacks
import matplotlib.pyplot as plt
from datetime import datetime

import CustomCallback
import numpy as np
import PIL

import tensorflow as tf
import tensorflow_datasets as tfds

import tensorboard


# Method for logging the training of a network, with the aim of visualising using tensorboard:

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# Naming convention for saved models:

MODEL_PREFIX = "models/"
MODEL_NAME = "model3.h5"


# Defining parameters batch size and the number of epochs:

batch_size = 64
epochs = 10


# Function to normalise the input images:

def resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    return (image, label)


# Initialises the dataset and the validation dataset:

ds = tfds.load('oxford_flowers102', split='train', shuffle_files=True)
ds_val = tfds.load('oxford_flowers102', split="validation", shuffle_files=True)


# Checking that the two datasets are indeed datasets:

assert isinstance(ds, tf.data.Dataset)
assert isinstance(ds_val, tf.data.Dataset)


# Normalising the pixel values of the input images:

ds = ds.map(lambda b: (b['image'] / 255, b['label']))
ds_val = ds_val.map(lambda a: (a['image'] / 255, a['label']))


# Running the normalisation function on the input images within the datasets:

ds_val = ds_val.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)

ds = ds.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)


# Initialising the model:

model = tf.keras.models.Sequential()


# Adding some random flip layers: (Input Shape - width, height, colour channels)

model.add(tf.keras.layers.RandomFlip("horizontal", input_shape=(256, 256, 3)))
model.add(tf.keras.layers.RandomFlip("vertical"))


# Adding some randomisation layers:

model.add(tf.keras.layers.RandomRotation(0.9))
model.add(tf.keras.layers.RandomZoom(0.3))
model.add(tf.keras.layers.RandomCrop(230, 230))
model.add(tf.keras.layers.RandomTranslation(0.3, 0.3))

# model.add(tf.keras.layers.RandomWidth(0.2))
# model.add(tf.keras.layers.RandomBrightness(0.1))


# Adding convolution layer and applying average pooling:

model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())

# model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())

model.add(tf.keras.layers.Conv2D(64, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())

# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv2D(128, (3, 3), 1, padding='same', activation='relu'))


# Adding a spacial dropout layer:

model.add(tf.keras.layers.SpatialDropout2D(0.4))


# Adding a flattening layer:

model.add(tf.keras.layers.Flatten())


# Adding dense, batch normalisation and dropout layers:

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.62))
model.add(tf.keras.layers.Dense(102, activation='softmax'))


# Compling the model:

model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


# Summarising the model:

model.summary()


# Printing the start time of the training process:

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time:", start_time)


# Beginning the training process:

hist = model.fit(ds, epochs=epochs, validation_data=ds_val, callbacks=[tensorboard_callback])


# Saves the current model to the program folders under the name in the brackets:

model.save(MODEL_PREFIX+MODEL_NAME)


# Printing the start and the end time of the training process:

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time:", start_time)
print("End Time:", current_time)


# Initialising a plot for the accuracy vs the value accuracy of the model:

plt.figure()


# Adding the accuracy and value accuracy to the plot:

plt.plot(hist.history['accuracy'], color='orange', label='training-accuracy')
plt.plot(hist.history['val_accuracy'], color='red', label='val_accuracy')


# Adding the loss and value loss to the plot:

# plt.plot(hist.history['loss'], color='teal', label='training-loss')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss')


# Changing the aesthetics of the plot:

plt.plot(hist.history['val_accuracy'], color='red', label='val_accuracy')
plt.title('Training Loss & Accuracy')
plt.legend(loc="upper left")


# Displaying the plot on the screen:

plt.show()