import keras.callbacks
import matplotlib.pyplot as plt
from datetime import datetime
import CustomCallback
import numpy as np
import PIL

import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 64
epochs = 256


now = datetime.now()
def resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    return (image, label)


ds = tfds.load('oxford_flowers102', split='train', shuffle_files=True)
ds_val = tfds.load('oxford_flowers102', split="validation", shuffle_files=True)

assert isinstance(ds, tf.data.Dataset)
assert isinstance(ds_val, tf.data.Dataset)

ds = ds.map(lambda b: (b['image'] / 255, b['label']))
ds_val = ds_val.map(lambda a: (a['image'] / 255, a['label']))

ds_val = ds_val.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)

ds = ds.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)

model = tf.keras.models.Sequential()

# Input Shape : width, height, colour channels
model.add(tf.keras.layers.RandomFlip("horizontal", input_shape=(256, 256, 3)))
model.add(tf.keras.layers.RandomFlip("vertical"))

model.add(tf.keras.layers.RandomRotation(0.9))
model.add(tf.keras.layers.RandomZoom(0.3))
model.add(tf.keras.layers.RandomCrop(230, 230))
model.add(tf.keras.layers.RandomTranslation(0.3, 0.3))

#model.add(tf.keras.layers.RandomWidth(0.2))

# model.add(tf.keras.layers.RandomBrightness(0.1))

model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())

model.add(tf.keras.layers.Conv2D(64, (3, 3), 1, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Conv2D(128, (3, 3), 1, padding='same', activation='relu'))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.62))

model.add(tf.keras.layers.Dense(102, activation='softmax'))

model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.summary()




logdir = "logs"
tensorboard_callback = CustomCallback.CustomCallback()

start_time = now.strftime("%H:%M:%S")
print("Start Time:", start_time)

hist = model.fit(ds, epochs=epochs, validation_data=ds_val)




now = datetime.now()
current_time = now.strftime("%H:%M:%S")

print("Start Time:", start_time)
print("End Time:", current_time)

model.save('models/model1')
plt.figure()

# plt.plot(hist.history['loss'], color='teal', label='training-loss')
plt.plot(hist.history['accuracy'], color='orange', label='training-accuracy')
# plt.plot(hist.history['val_loss'], color='blue', label='val_loss')
plt.plot(hist.history['val_accuracy'], color='red', label='val_accuracy')
plt.title('Training Loss & Accuracy')
plt.legend(loc="upper left")
plt.show()


