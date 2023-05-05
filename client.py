import os
import numpy as np
import tensorflow as tf
import flwr as fl

def load_data(prefix):
    X = np.loadtxt(os.path.join('UCI_HAR_Dataset', prefix, 'X_{}.txt'.format(prefix)))
    y = np.loadtxt(os.path.join('UCI_HAR_Dataset', prefix, 'y_{}.txt'.format(prefix))) - 1
    return X, y

def preprocess_data():
    X_train, y_train = load_data('train')
    X_test, y_test = load_data('test')

    y_train = tf.keras.utils.to_categorical(y_train, 6)
    y_test = tf.keras.utils.to_categorical(y_test, 6)

    return X_train, y_train, X_test, y_test

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(561,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

class HARClient(fl.client.Client):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], verbose=2)
        return self.get_parameters(), len(X_train)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=2)
        return loss, len(X_test)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess_data()
    model = create_model()
    client = HARClient(model)
    fl.client.start_client("0.0.0.0:8080", client)
