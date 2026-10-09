from __future__ import annotations

import logging
import time
from typing import Tuple

import keras
import numpy as np
from keras import Model, layers
from keras.applications import (
    ConvNeXtBase,
    DenseNet121,
    EfficientNetV2B0,
    EfficientNetV2B3,
    MobileNetV2,
    ResNet50,
)

from waste_sorting.config import Config
from waste_sorting.modeling.losses import focal_loss
from waste_sorting.modeling.preprocessing import BackbonePreprocess

logger = logging.getLogger(__name__)

BASE_MODELS = {
    "ResNet50": ResNet50,
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B3": EfficientNetV2B3,
    "MobileNetV2": MobileNetV2,
    "DenseNet121": DenseNet121,
    "ConvNeXtBase": ConvNeXtBase,
}


def _log_line(line, *args, **kwargs):
    # FIX: accepts extra args. Original: `print_fn=lambda x: ...` takes exactly one argument,
    # but Keras 3's summary may call print_fn(line, line_break=...) -> TypeError.
    logger.info(line)


def _load_backbone(arch: str, config: Config) -> Model:
    ctor = BASE_MODELS[arch]
    shape = (*config.img_size, 3)
    if not config.pretrained:
        return ctor(include_top=False, weights=None, input_shape=shape)

    last_err = None
    for attempt in range(1, 4):
        try:
            model = ctor(include_top=False, weights="imagenet", input_shape=shape)
            logger.info("Loaded pretrained ImageNet weights for %s", arch)
            return model
        except Exception as e:  # network / cache errors
            last_err = e
            logger.warning("Attempt %d/3 loading %s weights failed: %s", attempt, arch, e)
            time.sleep(2)
    if config.allow_random_init:
        logger.warning("Falling back to RANDOM init for %s (allow_random_init=True)", arch)
        return ctor(include_top=False, weights=None, input_shape=shape)
    raise RuntimeError(f"Could not load pretrained weights for {arch}") from last_err


def compile_model(model: Model, lr: float, config: Config) -> None:
    # FIX: one compile helper used by both phases.
    # Original: loss/optimizer/metrics were copy-pasted in two places and could drift apart.
    loss = (
        focal_loss(config.focal_loss_gamma, config.focal_loss_alpha)
        if config.use_focal_loss
        else "sparse_categorical_crossentropy"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=config.gradient_clip_value),
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        ],
    )


def unfreeze_top_layers(base_model: Model, percent_frozen: float) -> int:
    """Unfreeze the top (1 - percent_frozen) of the backbone, keeping BatchNorm frozen."""
    base_model.trainable = True
    cutoff = int(len(base_model.layers) * percent_frozen)
    for i, layer in enumerate(base_model.layers):
        # FIX (new): BatchNorm layers stay frozen during fine-tuning.
        # Original: every layer past the cutoff, BatchNorm included, was set trainable.
        # With a small learning rate and small batches, retraining BN scale/shift is a
        # common cause of fine-tuning accuracy collapsing right after unfreezing.
        layer.trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
    n = sum(1 for l in base_model.layers if l.trainable)
    logger.info("Unfroze %d/%d backbone layers", n, len(base_model.layers))
    return n


def build_model(arch: str, num_classes: int, config: Config) -> Tuple[Model, Model]:
    if arch not in BASE_MODELS:
        raise ValueError(f"Unknown architecture {arch!r}. Options: {sorted(BASE_MODELS)}")
    logger.info("Building %s...", arch)

    base_model = _load_backbone(arch, config)
    base_model.trainable = False

    inputs = keras.Input(shape=(*config.img_size, 3), name="image")  # raw 0-255 pixels
    x = BackbonePreprocess(arch, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout_rate * 0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout_rate * 0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name=arch)

    compile_model(model, config.learning_rate, config)

    model.summary(print_fn=_log_line)
    total = model.count_params()
    # FIX: np.prod(w.shape). Original: keras.backend.count_params(w), removed in Keras 3.
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    logger.info("Total params: %s | trainable: %s | frozen: %s", f"{total:,}", f"{trainable:,}", f"{total - trainable:,}")
    return model, base_model
