import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomRotation(0.2)])


def augment_img(x, y):
    x = data_augmentation(tf.expand_dims(x, -1))
    x = tf.squeeze(x, -1)
    return x, y


(x_train, _), (x_test, _) = fashion_mnist.load_data()

# sample_img = x_test[0]
# plt.imshow(sample_img, cmap='gray')
# plt.axis('off')
# plt.savefig('sample_img.png', bbox_inches='tight', pad_inches=0)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Jeśli używamy warstw konwolucyjnych
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train))
train_ds = train_ds.map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test))
test_ds = test_ds.batch(32)

latent_dim = 64


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(64, (3, 3), activation="sigmoid"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, (3, 3), activation="tanh"),
                tf.keras.layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(latent_dim, activation="relu"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [layers.Dense(784, activation="sigmoid"), layers.Reshape((28, 28))]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
autoencoder.fit(train_ds, epochs=10, shuffle=True, validation_data=test_ds)

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

autoencoder.encoder.save("encoder.keras")
autoencoder.decoder.save("decoder.keras")

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
