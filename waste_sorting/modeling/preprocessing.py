"""Backbone-specific input scaling, baked into the model.

Importing this module REGISTERS `BackbonePreprocess`, which keras.models.load_model needs.
"""
from __future__ import annotations

import keras
import numpy as np
from keras import ops

_CAFFE_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")
_TORCH_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
_TORCH_STD = np.array([0.229, 0.224, 0.225], dtype="float32")

# Backbones not listed (EfficientNetV2, ConvNeXt) include their own rescaling and expect 0-255.
_MODE_BY_ARCH = {"ResNet50": "caffe", "MobileNetV2": "tf", "DenseNet121": "torch"}


# FIX: this layer is new. It applies each backbone's correct preprocessing to raw 0-255 input.
# Original: every model received x/255 in [0,1], but
#   - ResNet50 expects BGR mean-subtracted pixels,
#   - MobileNetV2 expects [-1, 1],
#   - DenseNet121 expects (x/255 - mean)/std,
#   - EfficientNetV2 has its own Rescaling and expects [0,255] (so [0,1] looked almost black).
# Pretrained features were being fed out-of-distribution inputs, capping accuracy.
@keras.saving.register_keras_serializable(package="waste")
class BackbonePreprocess(keras.layers.Layer):
    def __init__(self, arch: str, **kwargs):
        super().__init__(**kwargs)
        self.arch = arch
        self.mode = _MODE_BY_ARCH.get(arch)

    def call(self, x):
        x = ops.cast(x, "float32")
        if self.mode == "caffe":
            return ops.flip(x, axis=-1) - ops.convert_to_tensor(_CAFFE_MEAN_BGR)
        if self.mode == "tf":
            return x / 127.5 - 1.0
        if self.mode == "torch":
            return (x / 255.0 - ops.convert_to_tensor(_TORCH_MEAN)) / ops.convert_to_tensor(_TORCH_STD)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg["arch"] = self.arch
        return cfg
