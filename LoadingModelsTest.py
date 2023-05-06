import tensorflow as tf
import tensorflow_datasets as tfds


# Naming convention for saved models, change to determine which model is loaded:

MODEL_PREFIX = "models/"
MODEL_NAME = "model2.h5"


# Defining the parameter batch size:

batch_size = 64


# Initialising the test dataset:

test_ds = tfds.load('oxford_flowers102', split='test', shuffle_files=True)


# Checking that the test dataset is indeed a dataset:

assert isinstance(test_ds, tf.data.Dataset)


# Normalising the pixel values of the test images:

test_ds = test_ds.map(lambda b: (b['image'] / 255, b['label']))


# Function to normalise the input images, same as the definition in main:

def resize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    return (image, label)


# Running the normalisation function on the test images within the dataset:

test_ds = test_ds.map(resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(
    tf.data.experimental.AUTOTUNE)


# Loading and summarising the model, name of the saved model to be loaded defined at the start of the program:

new_model = tf.keras.models.load_model(MODEL_PREFIX+MODEL_NAME)
new_model.summary()


# Checking the accuracy of the model:

loss, acc = new_model.evaluate(test_ds, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))