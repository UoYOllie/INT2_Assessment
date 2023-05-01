import tensorflow as tf
import pathlib
import scipy
import numpy

print("TensorFlow Version:", tf.__version__)

dataset_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
splits_url  = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"


archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
splits_archive = tf.keras.utils.get_file(origin=splits_url)
labels_archive = tf.keras.utils.get_file(origin=labels_url)

data_dir = pathlib.Path(archive).with_suffix('')
splits_dir = pathlib.Path(splits_archive).with_suffix('.mat')
labels_dir = pathlib.Path(labels_archive).with_suffix('.mat')

data_splits = scipy.io.loadmat(splits_dir)
data_labels = scipy.io.loadmat(labels_dir)

print(data_splits.keys())
print(data_labels.keys())
print(data_dir)

image_count = len(list(data_dir.glob('*.jpg')))


print(image_count)