import contextlib

import tensorflow as tf
from tensorflow import keras


def dense_model():
    inputs = tf.keras.Input(shape=(784,), name="image_floats")
    x = inputs
    x = keras.layers.Rescaling(1.0 / 255)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(10, activation="softmax")(x)
    outputs = x
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


MODELS = {"dense": dense_model}

DEFAULT_MODEL = "dense"

_SELECTED_MODEL = DEFAULT_MODEL


@contextlib.contextmanager
def use_model(model_name: str):
    global _SELECTED_MODEL

    if model_name not in MODELS:
        raise ValueError(f"Invalid model name: {model_name}")

    original_model = _SELECTED_MODEL
    try:
        _SELECTED_MODEL = model_name
        yield
    finally:
        _SELECTED_MODEL = original_model


def get_model(name=_SELECTED_MODEL):
    """Create a fresh model."""
    name = name or _SELECTED_MODEL
    if name not in MODELS:
        raise ValueError(
            f"Invalid model name '{name}', choose from: {', '.join(MODELS.keys())}"
        )
    model = MODELS[name]()
    return model


def get_compiled_model(name=_SELECTED_MODEL):
    """Create a fresh compiled model."""
    name = name or _SELECTED_MODEL
    model = get_model(name=name)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model
