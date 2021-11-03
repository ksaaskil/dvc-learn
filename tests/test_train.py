"""
Verify model before training, for example:

- Loss is strictly positive
- Loss decreases when training
- Predictions are as expected
- Trainable variables are updated
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pytest

from src.fashion.models import get_compiled_model
from src.fashion.io import get_training_data, label_to_class


def _train(model, train_dataset, val_dataset, epochs=2):
    tf.random.set_seed(33)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    return history


# @pytest.mark.skip(reason="Visualization")
def test_visualize():
    train_ds, _ = get_training_data(batch_size=8, train_size=500)

    batch = next(train_ds.take(1).as_numpy_iterator())

    x, y = batch
    for i in range(x.shape[0]):
        image = np.reshape(x[i, :], (28, 28))
        plt.imshow(image, cmap="gray_r")  # type: ignore
        clazz = label_to_class(y[i][0])
        plt.title(f"Class: {clazz}")
        plt.show()


@pytest.mark.pretrain
def test_model_training():
    model = get_compiled_model()
    train_ds, val_ds = get_training_data(batch_size=64, train_size=500)

    EPOCHS = 2
    history = _train(model, train_dataset=train_ds, val_dataset=val_ds, epochs=EPOCHS)

    assert len(model.trainable_weights) > 5
    assert len(model.non_trainable_weights) == 0

    loss = history.history["loss"]
    assert loss[1] < loss[0]
