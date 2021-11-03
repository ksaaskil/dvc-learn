"""
Verify model before training, for example:

- Loss is strictly positive
- Loss decreases when training
- Predictions are as expected
- Trainable variables are updated
"""
import numpy as np
from numpy.random import default_rng  # type: ignore
import pytest
import tensorflow as tf

from src.fashion.models import get_compiled_model


VALIDATION_RATIO = 0.20


def _get_datasets(batch_size: int, train_size: int = None):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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

    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train, val, test


def _train(model, train_dataset, val_dataset, epochs=2):
    tf.random.set_seed(33)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return history


@pytest.mark.pretrain
def test_model_training():
    model = get_compiled_model()
    train_ds, val_ds, _ = _get_datasets(batch_size=64, train_size=500)

    EPOCHS = 2
    history = _train(model, train_dataset=train_ds, val_dataset=val_ds, epochs=EPOCHS)

    assert len(model.trainable_weights) > 5
    assert len(model.non_trainable_weights) == 0

    loss = history.history["loss"]
    assert loss[1] < loss[0]
