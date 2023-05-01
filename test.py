import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

ds = tfds.load('oxford_flowers102', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)

# builder = tfds.builder('mnist')
# builder.download_and_prepare()
# ds = builder.as_dataset(split='train', shuffle_files=True)

print(ds)