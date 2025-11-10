import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
n_train = x_train.shape[0]
n_test = x_test.shape[0]
x_train_flat = x_train.reshape(n_train, -1)
x_test_flat = x_test.reshape(n_test, -1)

model = RandomForestClassifier(random_state=42)
model.fit(x_train_flat, y_train)

y_pred = model.predict(x_test_flat)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)


