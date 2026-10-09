from __future__ import annotations

from typing import Any, Dict, List

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

from waste_sorting.config import Config

_rng = np.random.default_rng(0)


def predict_with_tta(model: keras.Model, x_batch, config: Config) -> np.ndarray:
    """Class probabilities for one batch (0-255 pixels), optionally with test-time augmentation."""
    probs: List[np.ndarray] = [model(x_batch, training=False).numpy()]
    if config.use_tta:
        for _ in range(config.tta_steps - 1):
            aug = x_batch
            if _rng.random() > 0.5:
                aug = tf.image.flip_left_right(aug)
            # FIX: clip to [0, 255]. Original clipped to [0, 1], which would have destroyed
            # the image the moment inputs became raw 0-255 pixels.
            aug = tf.clip_by_value(aug * float(_rng.uniform(0.9, 1.1)), 0.0, 255.0)
            probs.append(model(aug, training=False).numpy())
    return np.mean(probs, axis=0)


def evaluate_model(model: keras.Model, test_ds, class_names: List[str], config: Config) -> Dict[str, Any]:
    y_true, y_prob = [], []
    # FIX: one batched forward call per batch via model(x).
    # Original: model.predict(x_batch) per batch (and per TTA pass) - each predict() call
    # rebuilds a data adapter and is far slower than calling the model directly.
    for x, y in tqdm(test_ds, desc="Evaluating", leave=False):
        y_true.append(y.numpy())
        y_prob.append(predict_with_tta(model, x, config))
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    # FIX: pass labels= explicitly everywhere.
    # Original: classification_report(target_names=...) raises ValueError if any class is
    # absent from both y_true and y_pred (e.g. a tiny test split); the confusion matrix
    # would also silently shrink and no longer line up with class_names.
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "cm": cm,
        "report": report,
        "y_pred": y_pred,
        "y_true": y_true,
        "y_pred_proba": y_prob,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
    }
