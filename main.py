import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    features = np.asarray(features).astype('float32')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(46),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.fit(features, labels, epochs=10)


if __name__ == "__main__":
    main()
