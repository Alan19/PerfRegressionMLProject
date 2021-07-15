
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

def main():

    data = pd.read_csv('data.csv')

    data.pop('Commit A')
    data.pop('Commit B')
    data.pop('Benchmark')
    data.pop('Top Chg by Instr >= X%')

    features = data.copy()
    labels = features.pop('Hit/Dismiss')

    labels_train = labels.tolist()

    labels_test = []

    for i in range(5500, 6383).__reversed__():
        labels_test.append(labels_train.pop(i))

    labels_test = np.asarray(labels_test)
    labels_train = np.asarray(labels_train)

    features = np.asarray(features).astype('float32')

    train = features.tolist()

    test = []

    for i in range(5500, 6383).__reversed__():
        test.append(train.pop(i))

    train = np.asarray(train).astype('float32')
    test = np.asarray(test).astype('float32')

    normalize = tf.keras.layers.experimental.preprocessing.Normalization()
    normalize.compile()
    normalize.adapt(train)

    print(normalize.count_params())
    print(train)

    model = tf.keras.models.Sequential([
        normalize,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  metrics=['accuracy'])

    model.fit(train, labels_train, epochs=1)

    print("\npredict\n")

    predictions = model.predict(test, batch_size=10, verbose=1)

    print(predictions)

if __name__ == "__main__":
    main()
