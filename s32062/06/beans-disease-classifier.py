import keras
import numpy as np
from keras import layers
import keras_tuner
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import random

from tensorflow.python.ops.gen_batch_ops import Batch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (500, 500)
BATCH_SIZE = 32

def preprocess(example_img, example_label):
    img = tf.cast(example_img, tf.float32) / 255.0

    return img, example_label

def create_model(activation='relu', optimizer='adam'):
    model = keras.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Flatten(),
        layers.Dense(64, activation=activation),
        layers.Dense(32),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def make_pipeline(ds, training=False):
    ds = ds.map(preprocess)

    if training:
        ds = ds.shuffle(1000, seed=SEED)

    ds = ds.batch(BATCH_SIZE)

    return ds


def main():
    splits = tfds.load('beans', split=['train', 'validation', 'test'], as_supervised=True, read_config=tfds.ReadConfig(shuffle_seed=SEED))
    ds_train, ds_validation, ds_test = splits

    ds_train = make_pipeline(ds_train, training=True)
    ds_validation = make_pipeline(ds_validation)
    ds_test = make_pipeline(ds_test)

    model = create_model()
    model.fit(ds_train, validation_data=ds_validation, epochs=5)
    model.evaluate(ds_test)



if __name__ == "__main__":
    main()
    # ds_train_prep = ds_train / 255.0
    #
    # model = create_model()
    # model.fit(x=ds_train_prep, y=ds_test, epochs=5)

    # print(info.splits["train"].num_examples)
    # print(info.features["label"].num_classes)

    # ds = tfds.load("beans", split="train", as_supervised=True)
    # for image, label in ds.take(1):
    #     print("Label:", label.numpy())
    #     plt.imshow(image.numpy())
    #     # plt.axis("off")
    #     plt.show()

