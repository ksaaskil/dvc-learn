import os
import gzip

import dvc.api
import numpy as np
from numpy.random import default_rng  # type: ignore
import tensorflow as tf

VALIDATION_RATIO = 0.20

FASHION_PATH = "data/fashion"


def data_url() -> str:
    """ Returns array of files in the directory, such as
    ```
    [
        {"md5": "bef4ecab320f06d8554ea6380940ec79", "relpath": "t10k-images-idx3-ubyte.gz"},
        {"md5": "bb300cfdad3c16e7a12a480ee83cd310", "relpath": "t10k-labels-idx1-ubyte.gz"},
        {"md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d", "relpath": "train-images-idx3-ubyte.gz"},
        {"md5": "25c81989df183df01b3e8a0aad5dffbe", "relpath": "train-labels-idx1-ubyte.gz"}
    ]
    ```
    """
    return dvc.api.get_url(path="data/fashion")


def resolve_url(path: str) -> str:
    return dvc.api.get_url(path=path)


def load_mnist(kind="train"):
    """
    Load MNIST data from `path`
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    labels_path = os.path.join(FASHION_PATH, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(FASHION_PATH, "%s-images-idx3-ubyte.gz" % kind)

    labels_path = resolve_url(labels_path)
    images_path = resolve_url(images_path)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def get_training_data(batch_size: int, train_size: int = None):
    x_train, y_train = load_mnist(kind="train")

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
    x_test, y_test = load_mnist(kind="t10k")
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
