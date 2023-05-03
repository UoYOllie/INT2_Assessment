import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

import tensorflow_datasets as tfds

ds = tfds.load('oxford_flowers102', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)

ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
for example in ds.take(1):
  image, label = example["image"], example["label"]
  print(image, label)



# plt.figure()
# plt.imshow(ds.images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# builder = tfds.builder('mnist')
# builder.download_and_prepare()
# ds = builder.as_dataset(split='train', shuffle_files=True)

