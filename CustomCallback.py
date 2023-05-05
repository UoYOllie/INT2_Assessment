import tensorflow as tf
from keras.utils import io_utils
from tensorflow import keras
import matplotlib.pyplot as plt


class CustomCallback(keras.callbacks.Callback):


    def on_train_begin(self, logs={}):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

        self.metrics = {"val_accuracy" : [],
                        "accuracy" : []}

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))

        if epoch == 0:
            self.f, self.axs = plt.subplots(1, len(self.metrics), figsize=(15, 5))

        #clear_output(wait=True)

        for i, metric in enumerate(self.metrics):
            self.axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)

            self.axs[i].legend()
            self.axs[i].grid()

        plt.tight_layout()

        if (epoch == 0):
            plt.ion()
            plt.show()

        if (plt.isinteractive() == False):
            plt.ion()

        # plt.pause(0.01)
        # plt.show()


    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass
