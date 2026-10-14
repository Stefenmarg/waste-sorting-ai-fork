from __future__ import annotations

from keras import ops


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for SPARSE (integer) labels and softmax outputs."""

    # FIX: rewritten with keras.ops and one-hot conversion of integer labels.
    # Original: used keras.backend.log / pow / clip (removed in Keras 3, which TF>=2.16
    # ships) and multiplied y_true * log(y_pred) assuming ONE-HOT labels, while the dataset
    # yields integer labels -> shape error as soon as use_focal_loss=True.
    def loss_fn(y_true, y_pred):
        eps = 1e-7
        y_pred = ops.clip(ops.cast(y_pred, "float32"), eps, 1.0 - eps)
        y_true = ops.one_hot(ops.cast(ops.reshape(y_true, (-1,)), "int32"), y_pred.shape[-1])
        ce = -y_true * ops.log(y_pred)
        weight = alpha * y_true * ops.power(1.0 - y_pred, gamma)
        return ops.sum(weight * ce, axis=-1)

    return loss_fn
