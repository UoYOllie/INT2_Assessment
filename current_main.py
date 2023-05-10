import keras.callbacks
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy


# Method for logging the training of a network, with the aim of visualising using tensorboard:
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# Naming convention for saved models, change to alter the name this model will be saved under:
MODEL_PREFIX = "models/" # Internal directory of models
MODEL_NAME = "name.h5"


# Defining parameters batch size and the number of epochs:
batch_size = 256
epochs = 4096

# Initialising the dataset and the validation dataset:
ds = tfds.load('oxford_flowers102', split='train', shuffle_files=True)
ds_val = tfds.load('oxford_flowers102', split="validation", shuffle_files=True)


# Checking that the two datasets are indeed datasets.
assert isinstance(ds, tf.data.Dataset)
assert isinstance(ds_val, tf.data.Dataset)


# Normalising the pixel values of the input images:
ds = ds.map(lambda b: (b['image'] / 255, b['label']))
ds_val = ds_val.map(lambda a: (a['image'] / 255, a['label']))


# Running the normalisation function on the input images within the datasets:
normalisation_layer = tf.keras.layers.CenterCrop(512, 512)
normalisation_layer_2 = tf.keras.layers.Resizing(128, 128, crop_to_aspect_ratio=True)

ds = ds.map(lambda a, b: (normalisation_layer(a), b))
ds_val = ds_val.map(lambda h, j: (normalisation_layer(h), j))

ds_val = ds_val.map(lambda d, e: (normalisation_layer_2(d), e), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)

ds = ds.map(lambda f, g: (normalisation_layer_2(f), g), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)



# Initialising the model:
model = tf.keras.models.Sequential()

# Adding some random flip layers: (Input Shape - width, height, colour channels)
model.add(tf.keras.layers.RandomFlip("horizontal", input_shape=(128, 128, 3)))
model.add(tf.keras.layers.RandomFlip("vertical"))

# Adding some randomisation layers:
model.add(tf.keras.layers.RandomRotation(1))
model.add(tf.keras.layers.RandomZoom(0.4))
model.add(tf.keras.layers.RandomCrop(230, 230))
model.add(tf.keras.layers.RandomContrast(0.25))


# Adding convolution layer and applying average pooling:
model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

# Adding a spacial dropout layer:
model.add(tf.keras.layers.SpatialDropout2D(0.12))


# Adding a flattening layer:
model.add(tf.keras.layers.Flatten())

# Adding dense, batch normalisation and dropout layers:
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.75))
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


# Saves the current model to the program folders under the name defined at the start of the program:
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


# Changing the aesthetics of the plot:
plt.plot(hist.history['val_accuracy'], color='red', label='val_accuracy')
plt.title('Training Loss & Accuracy')
plt.legend(loc="upper left")


# Displaying the plot on the screen:
plt.show()
