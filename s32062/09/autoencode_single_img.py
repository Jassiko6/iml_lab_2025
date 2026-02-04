from pathlib import Path

from keras.src.utils import img_to_array
from keras_tuner.src.backend.io import tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import argparse

if __name__ == "__main__":
    encoder = load_model("encoder.keras")
    decoder = load_model("decoder.keras")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=Path)

    args = parser.parse_args()

    img_path = str(args.file_path)
    img = tf.keras.utils.load_img(img_path, target_size=(28, 28))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    img_gray = tf.image.rgb_to_grayscale(img_array)
    encoded_img = encoder.predict(img_gray)
    decoded_img = decoder.predict(encoded_img)

    plt.imshow(decoded_img[0], cmap="gray")
    plt.axis("off")
    plt.savefig("result.png", bbox_inches="tight", pad_inches=0)

    print(encoded_img)
