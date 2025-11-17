import keras_tuner as kt
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
import keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def flatten(train, test):
    return train.reshape((train.shape[0], -1)), test.reshape((test.shape[0], -1))


def build_base_model():
    model = RandomForestClassifier(random_state=42)

    return model

def build_dnn_model(hp: kt.HyperParameters):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

def train_and_eval_base_model(model, x_train_flat, y_train, x_test_flat, y_test):
    model.fit(x_train_flat, y_train)

    y_proba = model.predict_proba(x_test_flat)
    y_pred_from_proba = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_test, y_pred_from_proba)
    ce = log_loss(y_test, y_proba, labels=model.classes_)

    print("Accuracy:", acc)
    print("Log loss:", ce)

    return y_pred_from_proba

def train_and_evaluate_dnn(model, train_ds, test_ds, epochs):
    model.fit(train_ds, epochs=epochs)
    model.evaluate(test_ds, verbose=2)

def main():
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x[:-10000]
    x_val = x[-10000:]
    y_train = y[:-10000]
    y_val = y[-10000:]

    x_base_all = x
    x_base_all, x_base_test = flatten(x_base_all, x_test)
    x_base_all = x_base_all / 255.0
    x_base_test = x_base_test / 255.0

    x_train_dnn = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_val_dnn = np.expand_dims(x_val, -1).astype("float32") / 255.0
    x_test_dnn = np.expand_dims(x_test, -1).astype("float32") / 255.0

    # num_classes = 10
    # y_train_dnn = keras.utils.to_categorical(y_train, num_classes)
    # y_val_dnn = keras.utils.to_categorical(y_val, num_classes)

    x_dnn_all = np.concatenate((x_train_dnn, x_val_dnn))
    y_dnn_all = np.concatenate((y_train, y_val))

    print("--Base model--")
    base_model = build_base_model()
    y_pred_base = train_and_eval_base_model(base_model, x_base_all, y, x_base_test, y_test)

    print("--DNN model--")
    build_dnn_model(kt.HyperParameters())

    tuner = kt.RandomSearch(
        hypermodel=build_dnn_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="tuner_runs",
        project_name="05-dnn",
    )

    tuner.search_space_summary()
    tuner.search(x_train_dnn, y_train, epochs=5, validation_data=(x_val_dnn, y_val), batch_size=32)

    models = tuner.get_best_models(num_models=2)
    best_model = models[0]
    best_model.summary()

    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = build_dnn_model(best_hp)

    model.fit(x=x_dnn_all, y=y_dnn_all, epochs=5)
    y_logits_dnn = model.predict(x_test_dnn)
    y_pred_dnn = np.argmax(y_logits_dnn, axis=1)

    print("Classification report - Baseline")
    print(classification_report(y_test, y_pred_base, output_dict=True))

    print("Classification report - DNN")
    print(classification_report(y_test, y_pred_dnn, output_dict=True))

    print("Best parameters:")
    print(best_hp.values)
if __name__ == "__main__":
    main()