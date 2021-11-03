import os
import gzip

import numpy as np
from numpy.random import default_rng  # type: ignore
import tensorflow as tf

VALIDATION_RATIO = 0.20


def load_mnist(path, kind="train"):
    """
    Load MNIST data from `path`
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def get_training_data(batch_size: int, train_size: int = None):
    x_train, y_train = load_mnist("data/fashion", kind="train")

    x_train = np.reshape(x_train, [-1, 784]).astype(np.uint8)
    y_train = np.reshape(y_train, [-1, 1]).astype(np.uint8)
    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000, 1)

    n_train = len(y_train) if train_size is None else train_size

    rng = default_rng(seed=42)

    train_idx = rng.choice(
        range(len(y_train)), size=int(n_train * (1.0 - VALIDATION_RATIO)), replace=False
    )

    # Available indices without training data indices
    val_list = np.delete(range(len(y_train)), train_idx)
    val_idx = rng.choice(val_list, size=int(n_train * VALIDATION_RATIO), replace=False)

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(train_idx) > len(val_idx)

    shared_idx = list(set(train_idx) & set(val_idx))
    assert shared_idx == []

    train = tf.data.Dataset.from_tensor_slices(
        (x_train[train_idx], y_train[train_idx])
    ).batch(batch_size)

    val = tf.data.Dataset.from_tensor_slices(
        (x_train[val_idx], y_train[val_idx])
    ).batch(batch_size)

    return train, val


def get_test_data(batch_size: int):
    x_test, y_test = load_mnist("data/fashion", kind="t10k")
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return test


CLASSES = {
    "0": "T-shirt/top",
    "1": "Trouser",
    "2": "Pullover",
    "3": "Dress",
    "4": "Coat",
    "5": "Sandal",
    "6": "Shirt",
    "7": "Sneaker",
    "8": "Bag",
    "9": "Ankle boot",
}


def label_to_class(i: int) -> str:
    return CLASSES[str(i)]
