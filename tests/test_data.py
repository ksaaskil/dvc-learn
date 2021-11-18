"""
Verify model before training, for example:

- Loss is strictly positive
- Loss decreases when training
- Predictions are as expected
- Trainable variables are updated
"""
import numpy as np
import matplotlib.pyplot as plt

import pytest

from src.fashion.io import get_training_data, label_to_class


@pytest.mark.skip(reason="Visualization")
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
