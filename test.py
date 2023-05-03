
import tensorflow as tf
# import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np

# ds = tfds.load('oxford_flowers102', split='train', shuffle_files=True)
# assert isinstance(ds, tf.data.Dataset)

# builder = tfds.builder('mnist')
# builder.download_and_prepare()
# ds = builder.as_dataset(split='train', shuffle_files=True)

# .load and .builder serve the same function. tensorflow website says .load is easier


oxford_flowers102 = tf.keras.datasets.oxford_flowers102
(train_images, train_labels), (test_images, test_labels) = oxford_flowers102.load_data()

train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# x_train, x_test = x_train / 255.0, x_test / 255.0

